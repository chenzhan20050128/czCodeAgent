"""Conflict-checked compensation for managed file-tool changes."""

from __future__ import annotations

import base64
import binascii
import os
import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .domain import Event, FileSnapshot, SessionReducer, SessionState
from .store import RolloutStore
from .tools.filesystem import sha256_bytes


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
    ) -> None:
        root = Path(workspace).resolve(strict=True)
        if not root.is_dir():
            raise ValueError("workspace must be a directory")
        self.store = store
        self.state = state
        self.workspace = root

    def undo_turn(self, turn_id: str) -> UndoResult:
        if turn_id not in self.state.turns:
            raise UndoError(f"unknown turn: {turn_id}")
        previous = self.state.undo_results.get(turn_id)
        if previous is not None:
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

        completed: list[UndoFileResult] = []
        for item in eligible:
            try:
                if item.snapshot.existed_before:
                    _atomic_restore(
                        item.path,
                        item.before_bytes,
                        item.snapshot.before_mode,
                    )
                    completed.append(
                        UndoFileResult(str(item.path), "restored", "baseline restored")
                    )
                else:
                    item.path.unlink()
                    _fsync_directory(item.path.parent)
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
            if all(item.status in {"restored", "deleted"} for item in completed)
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
            error = self._path_error(path)
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
            if error is None:
                try:
                    current_stat = path.lstat()
                except FileNotFoundError:
                    error = "managed file is missing"
                except OSError as exception:
                    error = f"cannot inspect managed file: {exception}"
                else:
                    if stat.S_ISLNK(current_stat.st_mode):
                        error = "managed path is a symbolic link"
                    elif not stat.S_ISREG(current_stat.st_mode):
                        error = "managed path is not a regular file"
                    else:
                        try:
                            current = path.read_bytes()
                        except OSError as exception:
                            error = f"cannot read managed file: {exception}"
                        else:
                            if sha256_bytes(current) != snapshot.after_hash:
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

    def _path_error(self, path: Path) -> str | None:
        if not path.is_absolute():
            return "managed path is not canonical"
        try:
            common = os.path.commonpath((str(self.workspace), str(path)))
        except ValueError:
            return "managed path is outside workspace"
        if common != str(self.workspace):
            return "managed path is outside workspace"
        current = self.workspace
        try:
            parts = path.relative_to(self.workspace).parts
        except ValueError:
            return "managed path is outside workspace"
        for part in parts:
            current = current / part
            try:
                current_stat = current.lstat()
            except FileNotFoundError:
                continue
            except OSError as error:
                return f"cannot inspect managed path: {error}"
            if stat.S_ISLNK(current_stat.st_mode):
                return "managed path is a symbolic link"
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


def _atomic_restore(path: Path, content: bytes, mode: int | None) -> None:
    if mode is None:
        raise UndoError("original file snapshot has no mode")
    temp_path: Path | None = None
    try:
        descriptor, raw_temp = tempfile.mkstemp(prefix=f".{path.name}.mca-undo-", dir=path.parent)
        temp_path = Path(raw_temp)
        with os.fdopen(descriptor, "wb") as stream:
            os.fchmod(stream.fileno(), mode)
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_path, path)
        temp_path = None
        _fsync_directory(path.parent)
    finally:
        if temp_path is not None:
            try:
                temp_path.unlink()
            except FileNotFoundError:
                pass


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


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
