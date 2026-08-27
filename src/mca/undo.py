"""Conflict-checked compensation for managed file-tool changes."""

from __future__ import annotations

import base64
import binascii
import hashlib
import os
import secrets
import stat
from dataclasses import dataclass
from pathlib import Path

from .domain import (
    Event,
    FileSnapshot,
    SessionReducer,
    SessionState,
    reduce_undo_status,
)
from .store import RolloutStore
from .tools.filesystem import DEFAULT_MAX_FILE_BYTES, sha256_bytes


class UndoError(RuntimeError):
    """Raised when an undo request itself is invalid or cannot be recorded."""


class _UndoConflict(UndoError):
    """Raised when mutation-time compare-and-swap detects interference."""


@dataclass(frozen=True)
class UndoFileResult:
    path: str
    status: str
    detail: str

    def to_dict(self) -> dict[str, str]:
        return {"path": self.path, "status": self.status, "detail": self.detail}


@dataclass(frozen=True)
class UndoResult:
    turn_id: str
    status: str
    files: tuple[UndoFileResult, ...]


@dataclass(frozen=True)
class _EligibleFile:
    snapshot: FileSnapshot
    path: Path
    before_bytes: bytes


class ManagedUndo:
    """Undo one turn's managed writes after an all-file preflight."""

    def __init__(
        self,
        store: RolloutStore,
        state: SessionState,
        workspace: str | os.PathLike[str],
        *,
        max_file_bytes: int = DEFAULT_MAX_FILE_BYTES,
    ) -> None:
        root = Path(workspace).resolve(strict=True)
        if not root.is_dir():
            raise ValueError("workspace must be a directory")
        self.store = store
        self.state = state
        self.workspace = root
        if type(max_file_bytes) is not int or max_file_bytes < 1:
            raise ValueError("max_file_bytes must be a positive integer")
        self.max_file_bytes = max_file_bytes

    def undo_turn(self, turn_id: str) -> UndoResult:
        if turn_id not in self.state.turns:
            raise UndoError(f"unknown turn: {turn_id}")
        previous = self.state.undo_results.get(turn_id)
        if previous is not None and previous.payload.get("status") == "succeeded":
            return _result_from_event(previous)

        snapshots = sorted(
            (
                snapshot
                for (snapshot_turn, _), snapshot in self.state.file_snapshots.items()
                if snapshot_turn == turn_id
            ),
            key=lambda snapshot: snapshot.path,
        )
        eligible, preflight = self._preflight(snapshots)
        if any(item.status in {"conflict", "ineligible"} for item in preflight):
            normalized = tuple(
                item
                if item.status in {"conflict", "ineligible"}
                else UndoFileResult(item.path, "not_modified", "preflight aborted")
                for item in preflight
            )
            return self._record(
                UndoResult(
                    turn_id,
                    reduce_undo_status(item.status for item in normalized),
                    normalized,
                )
            )

        completed: list[UndoFileResult] = [
            result
            for result in preflight
            if result.status in {"already_restored", "already_deleted"}
        ]
        for item in eligible:
            try:
                if item.snapshot.existed_before:
                    file_status = _atomic_restore(
                        self.workspace,
                        item.path,
                        item.before_bytes,
                        item.snapshot.before_mode,
                        expected_hash=item.snapshot.after_hash,
                        expected_mode=item.snapshot.after_mode,
                        operation_key=turn_id,
                        max_file_bytes=self.max_file_bytes,
                    )
                    completed.append(
                        UndoFileResult(
                            str(item.path),
                            file_status,
                            "baseline is already present"
                            if file_status == "already_restored"
                            else "baseline restored",
                        )
                    )
                else:
                    file_status = _verified_delete(
                        self.workspace,
                        item.path,
                        expected_hash=item.snapshot.after_hash,
                        expected_mode=item.snapshot.after_mode,
                        operation_key=turn_id,
                        max_file_bytes=self.max_file_bytes,
                    )
                    completed.append(
                        UndoFileResult(
                            str(item.path), file_status, "created file removed"
                        )
                    )
            except _UndoConflict as error:
                completed.append(
                    UndoFileResult(str(item.path), "conflict", str(error))
                )
            except Exception as error:
                completed.append(
                    UndoFileResult(
                        str(item.path),
                        "failed",
                        f"undo failed: {type(error).__name__}: {error}",
                    )
                )
        status = reduce_undo_status(item.status for item in completed)
        return self._record(UndoResult(turn_id, status, tuple(completed)))

    def undo(self, turn_id: str) -> UndoResult:
        return self.undo_turn(turn_id)

    def _preflight(
        self, snapshots: list[FileSnapshot]
    ) -> tuple[list[_EligibleFile], list[UndoFileResult]]:
        eligible: list[_EligibleFile] = []
        results: list[UndoFileResult] = []
        for snapshot in snapshots:
            path = Path(snapshot.path)
            error = self._path_error(snapshot.path)
            before_bytes: bytes | None = None
            if error is None:
                try:
                    before_bytes = base64.b64decode(
                        snapshot.before_bytes.encode("ascii"), validate=True
                    )
                except (UnicodeEncodeError, binascii.Error):
                    error = "snapshot before_bytes is not valid base64"
                else:
                    if len(before_bytes) > self.max_file_bytes:
                        error = "snapshot baseline exceeds size limit"
            if error is None and snapshot.after_hash is None:
                error = "snapshot has no successful after_hash"
            if error is None and snapshot.after_mode is None:
                error = "snapshot has no successful after_mode"
            if (
                error is None
                and snapshot.existed_before
                and snapshot.before_mode is None
            ):
                error = "original file snapshot has no mode"
            if error is None:
                quarantine_name = _deterministic_quarantine_name(
                    path.name,
                    operation_key=snapshot.turn_id,
                    path=path,
                    expected_hash=snapshot.after_hash,
                )
                try:
                    parent_fd = _open_parent_fd(self.workspace, path)
                    try:
                        quarantine = _read_optional_at(
                            parent_fd, quarantine_name, max_file_bytes=self.max_file_bytes
                        )
                        current_entry = _read_optional_at(
                            parent_fd, path.name, max_file_bytes=self.max_file_bytes
                        )
                    finally:
                        os.close(parent_fd)
                except UndoError as exception:
                    error = str(exception)
                except OSError as exception:
                    error = f"cannot inspect managed file: {exception}"
                else:
                    baseline_hash = sha256_bytes(before_bytes or b"")
                    if quarantine is not None:
                        quarantine_bytes, quarantine_stat = quarantine
                        if _is_quarantine_reservation(
                            quarantine_bytes, quarantine_stat
                        ) and _matches_state(
                            current_entry, snapshot.after_hash, snapshot.after_mode
                        ):
                            pass
                        elif (
                            sha256_bytes(quarantine_bytes) != snapshot.after_hash
                            or stat.S_IMODE(quarantine_stat.st_mode) != snapshot.after_mode
                        ):
                            error = "undo quarantine changed after the recorded write"
                        elif current_entry is None:
                            pass
                        elif (
                            snapshot.existed_before
                            and sha256_bytes(current_entry[0]) == baseline_hash
                            and stat.S_IMODE(current_entry[1].st_mode)
                            == snapshot.before_mode
                        ):
                            pass
                        else:
                            error = (
                                "managed target changed and exists alongside undo "
                                "quarantine"
                            )
                    elif current_entry is None:
                        if snapshot.existed_before:
                            error = "managed file is missing"
                        else:
                            results.append(
                                UndoFileResult(
                                    str(path),
                                    "already_deleted",
                                    "created file is already absent",
                                )
                            )
                            continue
                    else:
                        current, current_stat = current_entry
                        current_hash = sha256_bytes(current)
                        if snapshot.existed_before and current_hash == baseline_hash:
                            if stat.S_IMODE(current_stat.st_mode) == snapshot.before_mode:
                                results.append(
                                    UndoFileResult(
                                        str(path),
                                        "already_restored",
                                        "baseline is already present",
                                    )
                                )
                                continue
                            error = "managed file mode changed after the recorded write"
                        if current_hash != snapshot.after_hash:
                            error = "managed file changed after the recorded write"
                        elif stat.S_IMODE(current_stat.st_mode) != snapshot.after_mode:
                            error = "managed file mode changed after the recorded write"
            if error is not None:
                status = (
                    "conflict"
                    if "changed" in error or "missing" in error
                    else "ineligible"
                )
                results.append(UndoFileResult(str(path), status, error))
                continue
            assert before_bytes is not None
            eligible.append(_EligibleFile(snapshot, path, before_bytes))
            results.append(UndoFileResult(str(path), "eligible", "preflight passed"))
        return eligible, results

    def _path_error(self, raw_path: str) -> str | None:
        path = Path(raw_path)
        if not path.is_absolute() or os.path.normpath(raw_path) != raw_path:
            return "managed path is not canonical"
        try:
            common = os.path.commonpath((str(self.workspace), str(path)))
        except ValueError:
            return "managed path is outside workspace"
        if common != str(self.workspace):
            return "managed path is outside workspace"
        try:
            parts = path.relative_to(self.workspace).parts
        except ValueError:
            return "managed path is outside workspace"
        if not parts or any(part in {"", ".", ".."} for part in parts):
            return "managed path is not canonical"
        return None

    def _record(self, result: UndoResult) -> UndoResult:
        event = self.store.append(
            "undo_finished",
            {
                "turn_id": result.turn_id,
                "status": result.status,
                "files": [item.to_dict() for item in result.files],
            },
        )
        try:
            SessionReducer.apply(self.state, event)
        except Exception as error:
            raise UndoError(
                f"durable undo event {event.seq} could not be applied to state"
            ) from error
        return result


def _atomic_restore(
    workspace: Path,
    path: Path,
    content: bytes,
    mode: int | None,
    *,
    expected_hash: str | None,
    expected_mode: int | None,
    operation_key: str,
    max_file_bytes: int,
) -> str:
    if mode is None:
        raise UndoError("original file snapshot has no mode")
    if expected_hash is None:
        raise UndoError("snapshot has no successful after_hash")
    if expected_mode is None:
        raise UndoError("snapshot has no successful after_mode")
    try:
        parent_fd = _open_parent_fd(workspace, path)
    except OSError as error:
        raise _UndoConflict("managed parent changed during undo") from error
    temp_name: str | None = None
    quarantine_name = _deterministic_quarantine_name(
        path.name, operation_key, path, expected_hash
    )
    installed = False
    try:
        if _name_exists(parent_fd, quarantine_name):
            quarantine = _read_current_at(
                parent_fd, quarantine_name, max_file_bytes=max_file_bytes
            )
            current = _read_optional_at(
                parent_fd, path.name, max_file_bytes=max_file_bytes
            )
            if _is_quarantine_reservation(*quarantine) and _matches_state(
                current, expected_hash, expected_mode
            ):
                _require_same_entry(parent_fd, quarantine_name, quarantine[1])
                assert current is not None
                _require_same_entry(parent_fd, path.name, current[1])
                os.unlink(quarantine_name, dir_fd=parent_fd)
                _fsync_parent(parent_fd)
                _quarantine_target(parent_fd, path.name, quarantine_name)
            try:
                quarantined_stat = _require_current_state(
                    parent_fd,
                    quarantine_name,
                    expected_hash,
                    expected_mode,
                    max_file_bytes=max_file_bytes,
                )
            except Exception as error:
                raise _UndoConflict(
                    "undo quarantine changed after the recorded write; "
                    f"preserved {quarantine_name}"
                ) from error
            current = _read_optional_at(
                parent_fd, path.name, max_file_bytes=max_file_bytes
            )
            if current is not None:
                if (
                    sha256_bytes(current[0]) == sha256_bytes(content)
                    and stat.S_IMODE(current[1].st_mode) == mode
                ):
                    _require_same_entry(parent_fd, quarantine_name, quarantined_stat)
                    os.unlink(quarantine_name, dir_fd=parent_fd)
                    _fsync_parent(parent_fd)
                    return "already_restored"
                raise _UndoConflict(
                    "target exists alongside undo quarantine; concurrent target was "
                    f"preserved; recover managed content from {quarantine_name}"
                )
        else:
            _quarantine_target(parent_fd, path.name, quarantine_name)
            try:
                quarantined_stat = _require_current_state(
                    parent_fd,
                    quarantine_name,
                    expected_hash,
                    expected_mode,
                    max_file_bytes=max_file_bytes,
                )
            except Exception as error:
                detail, _ = _restore_quarantine(
                    parent_fd, quarantine_name, path.name
                )
                raise _UndoConflict(
                    f"managed file changed during undo; {detail}"
                ) from error

        try:
            _require_same_entry(parent_fd, quarantine_name, quarantined_stat)
        except Exception as error:
            raise _UndoConflict("quarantined file changed during undo") from error

        temp_name, descriptor = _create_exclusive_file(
            parent_fd, path.name, "restore", mode
        )
        with os.fdopen(descriptor, "wb") as stream:
            os.fchmod(stream.fileno(), mode)
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        _require_same_entry(parent_fd, quarantine_name, quarantined_stat)
        try:
            os.link(
                temp_name,
                path.name,
                src_dir_fd=parent_fd,
                dst_dir_fd=parent_fd,
                follow_symlinks=False,
            )
        except FileExistsError as error:
            raise _UndoConflict(
                "target was recreated during undo; concurrent target was preserved; "
                f"recover managed content from {quarantine_name}"
            ) from error
        installed = True
        os.unlink(temp_name, dir_fd=parent_fd)
        temp_name = None
        os.unlink(quarantine_name, dir_fd=parent_fd)
        _fsync_parent(parent_fd)
        return "restored"
    except _UndoConflict:
        raise
    except Exception as error:
        if not installed and _name_exists(parent_fd, quarantine_name):
            detail, removed = _restore_quarantine(
                parent_fd, quarantine_name, path.name
            )
            raise UndoError(f"{error}; {detail}") from error
        raise
    finally:
        if temp_name is not None:
            try:
                os.unlink(temp_name, dir_fd=parent_fd)
            except FileNotFoundError:
                pass
        os.close(parent_fd)


def _verified_delete(
    workspace: Path,
    path: Path,
    *,
    expected_hash: str | None,
    expected_mode: int | None,
    operation_key: str,
    max_file_bytes: int,
) -> str:
    if expected_hash is None:
        raise UndoError("snapshot has no successful after_hash")
    if expected_mode is None:
        raise UndoError("snapshot has no successful after_mode")
    try:
        parent_fd = _open_parent_fd(workspace, path)
    except OSError as error:
        raise _UndoConflict("managed parent changed during undo") from error
    quarantine_name = _deterministic_quarantine_name(
        path.name, operation_key, path, expected_hash
    )
    try:
        if _name_exists(parent_fd, quarantine_name):
            quarantine = _read_current_at(
                parent_fd, quarantine_name, max_file_bytes=max_file_bytes
            )
            current = _read_optional_at(
                parent_fd, path.name, max_file_bytes=max_file_bytes
            )
            if _is_quarantine_reservation(*quarantine) and _matches_state(
                current, expected_hash, expected_mode
            ):
                _require_same_entry(parent_fd, quarantine_name, quarantine[1])
                assert current is not None
                _require_same_entry(parent_fd, path.name, current[1])
                os.unlink(quarantine_name, dir_fd=parent_fd)
                _fsync_parent(parent_fd)
                _quarantine_target(parent_fd, path.name, quarantine_name)
            try:
                quarantined_stat = _require_current_state(
                    parent_fd,
                    quarantine_name,
                    expected_hash,
                    expected_mode,
                    max_file_bytes=max_file_bytes,
                )
            except Exception as error:
                raise _UndoConflict(
                    "undo quarantine changed after the recorded write; "
                    f"preserved {quarantine_name}"
                ) from error
            if _name_exists(parent_fd, path.name):
                raise _UndoConflict(
                    "target exists alongside undo quarantine; concurrent target was "
                    f"preserved; recover managed content from {quarantine_name}"
                )
        else:
            _quarantine_target(parent_fd, path.name, quarantine_name)
            try:
                quarantined_stat = _require_current_state(
                    parent_fd,
                    quarantine_name,
                    expected_hash,
                    expected_mode,
                    max_file_bytes=max_file_bytes,
                )
            except Exception as error:
                detail, _ = _restore_quarantine(
                    parent_fd, quarantine_name, path.name
                )
                raise _UndoConflict(
                    f"managed file changed during undo; {detail}"
                ) from error
        try:
            _require_same_entry(parent_fd, quarantine_name, quarantined_stat)
        except Exception as error:
            raise _UndoConflict("quarantined file changed during undo") from error
        if _name_exists(parent_fd, path.name):
            raise _UndoConflict(
                "target was recreated during undo; concurrent target was preserved; "
                f"recover managed content from {quarantine_name}"
            )
        os.unlink(quarantine_name, dir_fd=parent_fd)
        _fsync_parent(parent_fd)
        return "deleted"
    finally:
        os.close(parent_fd)


def _deterministic_quarantine_name(
    target_name: str, operation_key: str, path: Path, expected_hash: str | None
) -> str:
    if not operation_key or expected_hash is None:
        raise UndoError("deterministic undo quarantine requires operation identity")
    digest = hashlib.sha256(
        b"\0".join(
            (
                operation_key.encode("utf-8"),
                str(path).encode("utf-8"),
                expected_hash.encode("ascii"),
            )
        )
    ).hexdigest()[:16]
    return f".{target_name}.mca-undo-{digest}"


def _is_quarantine_reservation(content: bytes, file_stat: os.stat_result) -> bool:
    return not content and stat.S_IMODE(file_stat.st_mode) == 0o600


def _matches_state(
    current: tuple[bytes, os.stat_result] | None,
    expected_hash: str | None,
    expected_mode: int | None,
) -> bool:
    return (
        current is not None
        and expected_hash is not None
        and expected_mode is not None
        and sha256_bytes(current[0]) == expected_hash
        and stat.S_IMODE(current[1].st_mode) == expected_mode
    )


def _quarantine_target(parent_fd: int, target_name: str, quarantine_name: str) -> None:
    try:
        descriptor = os.open(
            quarantine_name,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0),
            0o600,
            dir_fd=parent_fd,
        )
    except FileExistsError as error:
        raise _UndoConflict(
            f"undo quarantine already exists: {quarantine_name}"
        ) from error
    placeholder_stat = os.fstat(descriptor)
    os.close(descriptor)
    try:
        os.rename(
            target_name,
            quarantine_name,
            src_dir_fd=parent_fd,
            dst_dir_fd=parent_fd,
        )
    except BaseException:
        try:
            _require_same_entry(parent_fd, quarantine_name, placeholder_stat)
        except (FileNotFoundError, _UndoConflict):
            pass
        else:
            os.unlink(quarantine_name, dir_fd=parent_fd)
        raise


def _restore_quarantine(
    parent_fd: int, quarantine_name: str, target_name: str
) -> tuple[str, bool]:
    try:
        os.link(
            quarantine_name,
            target_name,
            src_dir_fd=parent_fd,
            dst_dir_fd=parent_fd,
            follow_symlinks=False,
        )
    except FileExistsError:
        return (
            "concurrent target was preserved; recover managed content from "
            f"{quarantine_name}",
            False,
        )
    except OSError as error:
        return (
            f"recover managed content from {quarantine_name}; "
            f"link-back failed: {error}",
            False,
        )
    os.unlink(quarantine_name, dir_fd=parent_fd)
    _fsync_parent(parent_fd)
    return "original managed path was restored", True


def _create_exclusive_file(
    parent_fd: int, target_name: str, purpose: str, mode: int
) -> tuple[str, int]:
    for _ in range(128):
        candidate = _unused_name(parent_fd, target_name, purpose)
        try:
            descriptor = os.open(
                candidate,
                os.O_WRONLY
                | os.O_CREAT
                | os.O_EXCL
                | getattr(os, "O_NOFOLLOW", 0),
                mode,
                dir_fd=parent_fd,
            )
        except FileExistsError:
            continue
        return candidate, descriptor
    raise UndoError("could not allocate undo temporary file")


def _unused_name(parent_fd: int, target_name: str, purpose: str) -> str:
    for _ in range(128):
        candidate = f".{target_name}.mca-{purpose}-{secrets.token_hex(16)}"
        try:
            os.stat(candidate, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            return candidate
    raise UndoError(f"could not allocate undo {purpose} name")


def _name_exists(parent_fd: int, name: str) -> bool:
    try:
        os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        return False
    return True


def _require_same_entry(
    parent_fd: int, name: str, expected: os.stat_result
) -> None:
    current = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    if (current.st_dev, current.st_ino) != (expected.st_dev, expected.st_ino):
        raise _UndoConflict("quarantined file changed during undo")


def _open_parent_fd(workspace: Path, path: Path) -> int:
    """Open the parent chain without following symlinks (POSIX/macOS)."""

    try:
        parts = path.relative_to(workspace).parts
    except ValueError:
        raise UndoError("managed path is outside workspace") from None
    if not parts or any(part in {"", ".", ".."} for part in parts):
        raise UndoError("managed path is not canonical")
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(workspace, flags)
    try:
        for part in parts[:-1]:
            next_descriptor = os.open(part, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = next_descriptor
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _fsync_parent(parent_fd: int) -> None:
    os.fsync(parent_fd)


def _read_current(
    workspace: Path, path: Path, *, max_file_bytes: int
) -> tuple[bytes, os.stat_result]:
    parent_fd = _open_parent_fd(workspace, path)
    try:
        return _read_current_at(parent_fd, path.name, max_file_bytes=max_file_bytes)
    finally:
        os.close(parent_fd)


def _read_current_at(
    parent_fd: int, name: str, *, max_file_bytes: int
) -> tuple[bytes, os.stat_result]:
    descriptor = os.open(
        name, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0), dir_fd=parent_fd
    )
    try:
        file_stat = os.fstat(descriptor)
        if not stat.S_ISREG(file_stat.st_mode):
            raise UndoError("managed path is not a regular file")
        if file_stat.st_size > max_file_bytes:
            raise UndoError("managed file exceeds size limit")
        chunks: list[bytes] = []
        remaining = max_file_bytes + 1
        while remaining:
            chunk = os.read(descriptor, min(64 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        content = b"".join(chunks)
        if len(content) > max_file_bytes:
            raise UndoError("managed file exceeds size limit")
        return content, file_stat
    finally:
        os.close(descriptor)


def _read_optional_at(
    parent_fd: int, name: str, *, max_file_bytes: int
) -> tuple[bytes, os.stat_result] | None:
    try:
        return _read_current_at(
            parent_fd, name, max_file_bytes=max_file_bytes
        )
    except FileNotFoundError:
        return None


def _require_current_hash(
    parent_fd: int, name: str, expected_hash: str, *, max_file_bytes: int
) -> os.stat_result:
    content, file_stat = _read_current_at(
        parent_fd, name, max_file_bytes=max_file_bytes
    )
    if sha256_bytes(content) != expected_hash:
        raise UndoError("managed file changed during undo")
    return file_stat


def _require_current_state(
    parent_fd: int,
    name: str,
    expected_hash: str,
    expected_mode: int,
    *,
    max_file_bytes: int,
) -> os.stat_result:
    file_stat = _require_current_hash(
        parent_fd, name, expected_hash, max_file_bytes=max_file_bytes
    )
    if stat.S_IMODE(file_stat.st_mode) != expected_mode:
        raise UndoError("managed file mode changed during undo")
    return file_stat


def _result_from_event(event: Event) -> UndoResult:
    files = tuple(
        UndoFileResult(
            path=item["path"], status=item["status"], detail=item["detail"]
        )
        for item in event.payload["files"]
    )
    return UndoResult(
        turn_id=event.payload["turn_id"], status=event.payload["status"], files=files
    )


__all__ = ["ManagedUndo", "UndoError", "UndoFileResult", "UndoResult"]
