"""Conflict-checked compensation for managed file-tool changes."""

from __future__ import annotations

import base64
import binascii
import os
import secrets
import stat
from dataclasses import dataclass
from pathlib import Path

from .domain import Event, FileSnapshot, SessionReducer, SessionState
from .store import RolloutStore
from .tools.filesystem import DEFAULT_MAX_FILE_BYTES, sha256_bytes


class UndoError(RuntimeError):
    """Raised when an undo request itself is invalid or cannot be recorded."""


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
            return self._record(UndoResult(turn_id, "conflict", normalized))

        completed: list[UndoFileResult] = [
            result
            for result in preflight
            if result.status in {"already_restored", "already_deleted"}
        ]
        for item in eligible:
            try:
                if item.snapshot.existed_before:
                    _atomic_restore(
                        self.workspace,
                        item.path,
                        item.before_bytes,
                        item.snapshot.before_mode,
                        expected_hash=item.snapshot.after_hash,
                        max_file_bytes=self.max_file_bytes,
                    )
                    completed.append(
                        UndoFileResult(str(item.path), "restored", "baseline restored")
                    )
                else:
                    _verified_delete(
                        self.workspace,
                        item.path,
                        expected_hash=item.snapshot.after_hash,
                        max_file_bytes=self.max_file_bytes,
                    )
                    completed.append(
                        UndoFileResult(str(item.path), "deleted", "created file removed")
                    )
            except Exception as error:
                completed.append(
                    UndoFileResult(
                        str(item.path),
                        "failed",
                        f"undo failed: {type(error).__name__}: {error}",
                    )
                )
        status = (
            "succeeded"
            if all(
                item.status
                in {"restored", "deleted", "already_restored", "already_deleted"}
                for item in completed
            )
            else "partial"
        )
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
            if error is None and snapshot.after_hash is None:
                error = "snapshot has no successful after_hash"
            if (
                error is None
                and snapshot.existed_before
                and snapshot.before_mode is None
            ):
                error = "original file snapshot has no mode"
            if error is None:
                try:
                    current, current_stat = _read_current(
                        self.workspace, path, max_file_bytes=self.max_file_bytes
                    )
                except FileNotFoundError:
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
                except UndoError as exception:
                    error = str(exception)
                except OSError as exception:
                    error = f"cannot inspect managed file: {exception}"
                else:
                    current_hash = sha256_bytes(current)
                    baseline_hash = sha256_bytes(before_bytes or b"")
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
    max_file_bytes: int,
) -> None:
    if mode is None:
        raise UndoError("original file snapshot has no mode")
    if expected_hash is None:
        raise UndoError("snapshot has no successful after_hash")
    parent_fd = _open_parent_fd(workspace, path)
    temp_name: str | None = None
    try:
        _require_current_hash(
            parent_fd, path.name, expected_hash, max_file_bytes=max_file_bytes
        )
        for _ in range(128):
            candidate = f".{path.name}.mca-undo-{secrets.token_hex(8)}"
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
            temp_name = candidate
            break
        else:
            raise UndoError("could not allocate undo temporary file")
        with os.fdopen(descriptor, "wb") as stream:
            os.fchmod(stream.fileno(), mode)
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        _require_current_hash(
            parent_fd, path.name, expected_hash, max_file_bytes=max_file_bytes
        )
        os.replace(
            temp_name,
            path.name,
            src_dir_fd=parent_fd,
            dst_dir_fd=parent_fd,
        )
        temp_name = None
        _fsync_parent(parent_fd)
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
    max_file_bytes: int,
) -> None:
    if expected_hash is None:
        raise UndoError("snapshot has no successful after_hash")
    parent_fd = _open_parent_fd(workspace, path)
    try:
        inspected = _require_current_hash(
            parent_fd, path.name, expected_hash, max_file_bytes=max_file_bytes
        )
        current = os.stat(path.name, dir_fd=parent_fd, follow_symlinks=False)
        if (current.st_dev, current.st_ino) != (inspected.st_dev, inspected.st_ino):
            raise UndoError("managed file changed during undo")
        os.unlink(path.name, dir_fd=parent_fd)
        _fsync_parent(parent_fd)
    finally:
        os.close(parent_fd)


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


def _require_current_hash(
    parent_fd: int, name: str, expected_hash: str, *, max_file_bytes: int
) -> os.stat_result:
    content, file_stat = _read_current_at(
        parent_fd, name, max_file_bytes=max_file_bytes
    )
    if sha256_bytes(content) != expected_hash:
        raise UndoError("managed file changed during undo")
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
