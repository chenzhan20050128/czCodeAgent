"""Immutable whole-file versions and in-process mutation coordination."""

from __future__ import annotations

import hashlib
import os
import stat
import threading
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path


class FileVersionError(OSError):
    """Raised when a stable regular-file version cannot be captured."""


@dataclass(frozen=True)
class FileVersion:
    """One immutable whole-file identity used for compare-and-swap."""

    exists: bool
    sha256: str | None
    mode: int | None
    size: int | None
    device: int | None
    inode: int | None
    mtime_ns: int | None
    ctime_ns: int | None

    def __post_init__(self) -> None:
        if type(self.exists) is not bool:
            raise ValueError("exists must be a boolean")
        values = (
            self.sha256,
            self.mode,
            self.size,
            self.device,
            self.inode,
            self.mtime_ns,
            self.ctime_ns,
        )
        if not self.exists:
            if any(value is not None for value in values):
                raise ValueError("an absent file version has no metadata")
            return
        if not isinstance(self.sha256, str) or not self.sha256:
            raise ValueError("an existing file version requires sha256")
        if self.mode is None or type(self.mode) is not int or not 0 <= self.mode <= 0o7777:
            raise ValueError("an existing file version requires permission bits")
        for name in ("size", "device", "inode", "mtime_ns", "ctime_ns"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"an existing file version requires non-negative {name}")

    @classmethod
    def absent(cls) -> FileVersion:
        return cls(False, None, None, None, None, None, None, None)

    @classmethod
    def from_dict(cls, value: object) -> FileVersion:
        if not isinstance(value, Mapping):
            raise ValueError("file version must be an object")
        fields = {
            "exists",
            "sha256",
            "mode",
            "size",
            "device",
            "inode",
            "mtime_ns",
            "ctime_ns",
        }
        if set(value) != fields:
            raise ValueError("file version fields mismatch")
        return cls(**{field: value[field] for field in fields})

    def to_dict(self) -> dict[str, object]:
        return {
            "exists": self.exists,
            "sha256": self.sha256,
            "mode": self.mode,
            "size": self.size,
            "device": self.device,
            "inode": self.inode,
            "mtime_ns": self.mtime_ns,
            "ctime_ns": self.ctime_ns,
        }

    def display(self) -> str:
        if not self.exists:
            return "<absent>"
        return (
            f"{self.sha256} mode={self.mode:o} size={self.size} "
            f"dev={self.device} ino={self.inode} "
            f"mtime_ns={self.mtime_ns} ctime_ns={self.ctime_ns}"
        )


def capture_file_version(
    path: Path, *, max_file_bytes: int
) -> tuple[FileVersion, bytes]:
    """Read a regular file and return bytes tied to one stable fd identity."""

    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except FileNotFoundError:
        return FileVersion.absent(), b""
    except OSError as error:
        raise FileVersionError(f"cannot open file version: {path}") from error
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise FileVersionError("path is not a regular file")
        if before.st_size > max_file_bytes:
            raise FileVersionError("file exceeds size limit")
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
            raise FileVersionError("file exceeds size limit")
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    identity_before = _stat_identity(before)
    identity_after = _stat_identity(after)
    if identity_before != identity_after or after.st_size != len(content):
        raise FileVersionError("file changed while capturing version")
    try:
        named = path.stat(follow_symlinks=False)
    except OSError as error:
        raise FileVersionError("file changed while capturing version") from error
    if (named.st_dev, named.st_ino) != (after.st_dev, after.st_ino):
        raise FileVersionError("file changed while capturing version")
    return (
        FileVersion(
            exists=True,
            sha256=hashlib.sha256(content).hexdigest(),
            mode=stat.S_IMODE(after.st_mode),
            size=len(content),
            device=after.st_dev,
            inode=after.st_ino,
            mtime_ns=after.st_mtime_ns,
            ctime_ns=after.st_ctime_ns,
        ),
        content,
    )


def _stat_identity(value: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        stat.S_IMODE(value.st_mode),
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


@dataclass
class _PathQueue:
    condition: threading.Condition
    next_ticket: int = 0
    serving: int = 0
    users: int = 0


class DirectoryMutationLease:
    """Track parent usage and transfer failed-creator provenance safely."""

    def __init__(
        self, coordinator: FileMutationCoordinator, directories: tuple[str, ...]
    ) -> None:
        self._coordinator = coordinator
        self.directories = directories

    def successful_provenance(
        self, created: tuple[str, ...]
    ) -> tuple[str, ...]:
        return self._coordinator._successful_provenance(self.directories, created)

    def cleanup_failed(self, created: tuple[str, ...]) -> None:
        self._coordinator._cleanup_failed(self.directories, created)

    def create_directory(self, path: Path, mode: int) -> bool:
        return self._coordinator._create_directory(path, mode)


class FileMutationCoordinator:
    """Run same-canonical-path commits in FIFO arrival order."""

    def __init__(self) -> None:
        self._guard = threading.Lock()
        self._paths: dict[str, _PathQueue] = {}
        self._directory_users: dict[str, int] = {}
        self._unowned_directories: set[str] = set()
        self._claimed_directories: set[str] = set()

    @contextmanager
    def mutation(
        self, path: Path, parent_directories: tuple[Path, ...]
    ):
        canonical = str(path)
        if not path.is_absolute() or os.path.normpath(canonical) != canonical:
            raise ValueError("mutation path must be canonical and absolute")
        directories = tuple(str(directory) for directory in parent_directories)
        if any(
            not directory.is_absolute()
            or os.path.normpath(str(directory)) != str(directory)
            for directory in parent_directories
        ):
            raise ValueError("mutation parents must be canonical and absolute")
        with self._guard:
            for directory in directories:
                self._directory_users[directory] = (
                    self._directory_users.get(directory, 0) + 1
                )
        try:
            with self._turn(canonical):
                yield DirectoryMutationLease(self, directories)
        finally:
            with self._guard:
                for directory in directories:
                    users = self._directory_users[directory] - 1
                    if users:
                        self._directory_users[directory] = users
                    else:
                        del self._directory_users[directory]
                        self._unowned_directories.discard(directory)
                        self._claimed_directories.discard(directory)

    def _create_directory(self, path: Path, mode: int) -> bool:
        directory = str(path)
        with self._guard:
            try:
                os.mkdir(path, mode)
            except FileExistsError:
                return False
            self._unowned_directories.add(directory)
            return True

    def _successful_provenance(
        self, reserved: tuple[str, ...], created: tuple[str, ...]
    ) -> tuple[str, ...]:
        with self._guard:
            adopted = tuple(
                directory
                for directory in reserved
                if (
                    directory in self._unowned_directories
                    and directory not in self._claimed_directories
                    and _is_directory(directory)
                )
            )
            self._claimed_directories.update(adopted)
            self._unowned_directories.difference_update(adopted)
        return adopted

    def _cleanup_failed(
        self, reserved: tuple[str, ...], created: tuple[str, ...]
    ) -> None:
        with self._guard:
            pending = tuple(
                directory
                for directory in reserved
                if directory in self._unowned_directories
            )
            candidates = tuple(dict.fromkeys((*created, *pending)))
            for directory in sorted(
                candidates,
                key=lambda item: (len(Path(item).parts), len(item)),
                reverse=True,
            ):
                if directory in self._claimed_directories:
                    continue
                if self._directory_users.get(directory, 0) > 1:
                    continue
                try:
                    os.rmdir(directory)
                except OSError:
                    continue
                self._unowned_directories.discard(directory)

    @contextmanager
    def _turn(self, canonical: str):
        with self._guard:
            queue = self._paths.get(canonical)
            if queue is None:
                queue = _PathQueue(threading.Condition())
                self._paths[canonical] = queue
            queue.users += 1
        with queue.condition:
            ticket = queue.next_ticket
            queue.next_ticket += 1
            while ticket != queue.serving:
                queue.condition.wait()
        try:
            yield
        finally:
            with queue.condition:
                queue.serving += 1
                queue.condition.notify_all()
            with self._guard:
                queue.users -= 1
                if queue.users == 0 and self._paths.get(canonical) is queue:
                    del self._paths[canonical]


def _is_directory(path: str) -> bool:
    try:
        return stat.S_ISDIR(Path(path).lstat().st_mode)
    except OSError:
        return False


GLOBAL_FILE_MUTATION_COORDINATOR = FileMutationCoordinator()


__all__ = [
    "FileMutationCoordinator",
    "FileVersion",
    "FileVersionError",
    "DirectoryMutationLease",
    "GLOBAL_FILE_MUTATION_COORDINATOR",
    "capture_file_version",
]
