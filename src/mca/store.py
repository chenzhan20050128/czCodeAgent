"""Durable, single-writer JSONL storage for session events."""

from __future__ import annotations

import errno
import fcntl
import json
import os
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .domain import EVENT_FIELDS, DomainError, Event


class RolloutCorruptionError(RuntimeError):
    """Raised when a committed rollout record is invalid."""


class SessionLockedError(RuntimeError):
    """Raised when another writer already owns the session."""


def _validate_session_id(session_id: object) -> str:
    if not isinstance(session_id, str):
        raise ValueError("session_id must be a canonical UUID")
    try:
        parsed = uuid.UUID(session_id)
    except (ValueError, AttributeError, TypeError):
        raise ValueError("session_id must be a canonical UUID") from None
    if str(parsed) != session_id:
        raise ValueError("session_id must be a canonical UUID")
    return session_id


class RolloutStore:
    """Append-only event storage protected by an advisory writer lock."""

    def __init__(
        self,
        sessions_root: str | os.PathLike[str],
        session_id: str,
        *,
        create: bool,
    ) -> None:
        self.session_id = _validate_session_id(session_id)
        self.sessions_root = Path(sessions_root)
        self.path = self.sessions_root / f"{self.session_id}.jsonl"
        self._fd: int | None = None
        self._events: list[Event] = []

        self._prepare_directory()
        flags = os.O_RDWR | os.O_APPEND
        flags |= os.O_CREAT | os.O_EXCL if create else 0
        flags |= getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        try:
            fd = os.open(self.path, flags, 0o600)
        except FileExistsError:
            raise
        except OSError as error:
            if error.errno == errno.ELOOP:
                raise ValueError("rollout path must not be a symbolic link") from None
            raise

        self._fd = fd
        try:
            os.fchmod(fd, 0o600)
            self._acquire_lock()
            if create:
                os.fsync(fd)
                self._fsync_directory(self.sessions_root)
            self._events = self._read_and_repair_tail()
        except BaseException:
            self.close()
            raise

    @classmethod
    def create(
        cls, sessions_root: str | os.PathLike[str], session_id: str
    ) -> RolloutStore:
        return cls(sessions_root, session_id, create=True)

    @classmethod
    def open(
        cls, sessions_root: str | os.PathLike[str], session_id: str
    ) -> RolloutStore:
        return cls(sessions_root, session_id, create=False)

    def _prepare_directory(self) -> None:
        directory_existed = self.sessions_root.exists()
        self.sessions_root.mkdir(mode=0o700, parents=True, exist_ok=True)
        if not self.sessions_root.is_dir() or self.sessions_root.is_symlink():
            raise ValueError("sessions_root must be a real directory")
        os.chmod(self.sessions_root, 0o700)
        if not directory_existed:
            self._fsync_directory(self.sessions_root)
            self._fsync_directory(self.sessions_root.parent)

    @staticmethod
    def _fsync_directory(path: Path) -> None:
        flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        fd = os.open(path, flags)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)

    def _acquire_lock(self) -> None:
        assert self._fd is not None
        try:
            fcntl.flock(self._fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SessionLockedError(
                f"session {self.session_id} is already open for writing"
            ) from None

    def _read_all(self) -> bytes:
        assert self._fd is not None
        os.lseek(self._fd, 0, os.SEEK_SET)
        chunks: list[bytes] = []
        while True:
            chunk = os.read(self._fd, 64 * 1024)
            if not chunk:
                return b"".join(chunks)
            chunks.append(chunk)

    def _read_and_repair_tail(self) -> list[Event]:
        assert self._fd is not None
        data = self._read_all()
        records = data.splitlines(keepends=True)
        events: list[Event] = []
        offset = 0

        for index, record in enumerate(records):
            line_number = index + 1
            is_final = index == len(records) - 1
            has_newline = record.endswith((b"\n", b"\r"))
            body = record.rstrip(b"\r\n")
            try:
                document = json.loads(body.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                if is_final and not has_newline:
                    os.ftruncate(self._fd, offset)
                    os.fsync(self._fd)
                    break
                raise RolloutCorruptionError(
                    f"invalid JSON at line {line_number}: {error}"
                ) from error

            try:
                event = Event.from_dict(document)
            except DomainError as error:
                if self._is_incomplete_unterminated_event(
                    document, is_final=is_final, has_newline=has_newline
                ):
                    os.ftruncate(self._fd, offset)
                    os.fsync(self._fd)
                    break
                raise RolloutCorruptionError(
                    f"invalid event at line {line_number}: {error}"
                ) from error

            if event.session_id != self.session_id:
                raise RolloutCorruptionError(
                    f"session mismatch at line {line_number}"
                )
            expected_seq = len(events) + 1
            if event.seq != expected_seq:
                raise RolloutCorruptionError(
                    f"invalid sequence at line {line_number}: "
                    f"expected {expected_seq}, got {event.seq}"
                )
            events.append(event)
            offset += len(record)

            if is_final and not has_newline:
                self._write_all(b"\n")
                os.fsync(self._fd)

        return events

    @staticmethod
    def _is_incomplete_unterminated_event(
        document: object, *, is_final: bool, has_newline: bool
    ) -> bool:
        return (
            is_final
            and not has_newline
            and isinstance(document, dict)
            and not EVENT_FIELDS.issubset(document)
        )

    def append(
        self,
        event_or_type: Event | str,
        payload: Mapping[str, Any] | None = None,
    ) -> Event:
        self._require_open()
        next_seq = len(self._events) + 1
        if isinstance(event_or_type, Event):
            if payload is not None:
                raise TypeError("payload cannot be supplied with an Event")
            event = event_or_type
            if event.session_id != self.session_id:
                raise ValueError("event belongs to another session")
            if event.seq != next_seq:
                raise ValueError(
                    f"event sequence must be {next_seq}, got {event.seq}"
                )
        else:
            if not isinstance(event_or_type, str):
                raise TypeError("event must be an Event or event type string")
            event = Event.create(
                seq=next_seq,
                session_id=self.session_id,
                event_type=event_or_type,
                payload={} if payload is None else payload,
            )

        encoded = (
            json.dumps(
                event.to_dict(),
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
        ).encode("utf-8")
        self._write_all(encoded)
        assert self._fd is not None
        os.fsync(self._fd)
        self._events.append(event)
        return event

    def _write_all(self, data: bytes) -> None:
        self._require_open()
        assert self._fd is not None
        view = memoryview(data)
        while view:
            written = os.write(self._fd, view)
            if written == 0:
                raise OSError("rollout write made no progress")
            view = view[written:]

    def load(self) -> list[Event]:
        self._require_open()
        return list(self._events)

    def _require_open(self) -> None:
        if self._fd is None:
            raise ValueError("rollout store is closed")

    def close(self) -> None:
        if self._fd is None:
            return
        fd, self._fd = self._fd, None
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)

    def __enter__(self) -> RolloutStore:
        self._require_open()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()
