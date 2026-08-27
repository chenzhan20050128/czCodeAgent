"""Workspace-confined text-file tools with approval-safe preparation."""

from __future__ import annotations

import difflib
import hashlib
import os
import signal
import stat
import tempfile
import threading
from dataclasses import dataclass, field
from pathlib import Path, PureWindowsPath
from typing import Any

from .registry import ToolResult


DEFAULT_MAX_FILE_BYTES = 2 * 1024 * 1024
DEFAULT_MAX_READ_LINES = 200
DEFAULT_MAX_LIST_ENTRIES = 200


class FileToolError(ValueError):
    """Raised when a filesystem tool cannot safely satisfy a request."""


class PathSafetyError(FileToolError):
    """Raised when a requested path is outside the workspace policy."""


class FileConflictError(FileToolError):
    """Raised when a prepared write no longer matches the live filesystem."""


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


class WorkspaceResolver:
    """Resolve relative paths against one canonical workspace directory."""

    def __init__(self, workspace: str | os.PathLike[str]) -> None:
        root = Path(workspace).resolve(strict=True)
        if not root.is_dir():
            raise ValueError("workspace must be a directory")
        self.workspace = root

    def resolve_read(self, requested: str, *, must_exist: bool = True) -> Path:
        """Resolve a read path, following symlinks only when they remain inside."""

        relative = self._relative_path(requested)
        candidate = self.workspace.joinpath(relative)
        lexical = Path(os.path.abspath(candidate))
        self._require_inside(lexical)
        try:
            resolved = candidate.resolve(strict=must_exist)
        except (FileNotFoundError, NotADirectoryError, OSError) as error:
            raise FileToolError(f"path does not exist: {requested}") from error
        self._require_inside(resolved)
        return resolved

    def resolve_write(self, requested: str) -> Path:
        """Resolve a write path while rejecting every symlink component."""

        relative = self._relative_path(requested)
        candidate = Path(os.path.abspath(self.workspace.joinpath(relative)))
        self._require_inside(candidate)
        try:
            relative_parts = candidate.relative_to(self.workspace).parts
        except ValueError:
            raise PathSafetyError("path escapes workspace") from None
        current = self.workspace
        for part in relative_parts:
            current = current / part
            try:
                current_stat = current.lstat()
            except FileNotFoundError:
                continue
            except OSError as error:
                raise FileToolError(f"cannot inspect path: {requested}") from error
            if stat.S_ISLNK(current_stat.st_mode):
                raise PathSafetyError(
                    f"write path contains a symbolic link: {requested}"
                )
        return candidate

    def relative_display(self, path: Path) -> str:
        try:
            return path.relative_to(self.workspace).as_posix() or "."
        except ValueError:
            return str(path)

    def _relative_path(self, requested: str) -> Path:
        if not isinstance(requested, str) or not requested:
            raise PathSafetyError("path must be a non-empty relative path")
        if "\0" in requested:
            raise PathSafetyError("path must not contain NUL")
        path = Path(requested)
        if path.is_absolute() or PureWindowsPath(requested).is_absolute():
            raise PathSafetyError("absolute paths are not allowed")
        if any(part == ".." for part in path.parts):
            raise PathSafetyError("parent path components are not allowed")
        return path

    def _require_inside(self, path: Path) -> None:
        try:
            common = os.path.commonpath((str(self.workspace), str(path)))
        except ValueError:
            raise PathSafetyError("path escapes workspace") from None
        if common != str(self.workspace):
            raise PathSafetyError("path escapes workspace")


@dataclass(frozen=True)
class ExecutedFileChange:
    canonical_path: Path
    before_hash: str | None
    after_hash: str
    after_mode: int
    durability_warning: bool = False
    interruption_warning: bool = False


@dataclass(frozen=True)
class PreparedFileChange:
    """A side-effect-free file proposal that can later be revalidated."""

    canonical_path: Path
    before_hash: str | None
    diff: str
    existed_before: bool
    before_bytes: bytes
    before_mode: int | None
    proposed_bytes: bytes
    requested_path: str
    _resolver: WorkspaceResolver = field(repr=False, compare=False)
    _max_file_bytes: int = field(repr=False, compare=False)

    def execute(self) -> ExecutedFileChange:
        """Atomically apply the approved bytes if path and hash still match."""

        target = self._assert_unchanged()
        parent = target.parent
        if not parent.exists() or not parent.is_dir():
            raise FileToolError(f"parent directory does not exist: {parent}")
        temp_path: Path | None = None
        committed = False
        durability_warning = False
        interruption_warning = False
        try:
            descriptor, raw_temp_path = tempfile.mkstemp(
                prefix=f".{target.name}.mca-", dir=parent
            )
            temp_path = Path(raw_temp_path)
            mode = self.before_mode if self.before_mode is not None else 0o644
            with os.fdopen(descriptor, "wb") as stream:
                os.fchmod(stream.fileno(), mode)
                stream.write(self.proposed_bytes)
                stream.flush()
                os.fsync(stream.fileno())
            target = self._assert_unchanged()
            previous_mask: set[signal.Signals] | None = None
            try:
                if threading.current_thread() is threading.main_thread() and hasattr(
                    signal, "pthread_sigmask"
                ):
                    previous_mask = signal.pthread_sigmask(
                        signal.SIG_BLOCK, {signal.SIGINT}
                    )
                try:
                    os.replace(temp_path, target)
                except KeyboardInterrupt:
                    if target.exists() and sha256_bytes(target.read_bytes()) == sha256_bytes(
                        self.proposed_bytes
                    ):
                        committed = True
                        temp_path = None
                        interruption_warning = True
                    else:
                        raise
                else:
                    committed = True
                    temp_path = None
                if committed:
                    try:
                        _fsync_directory(parent)
                    except OSError:
                        durability_warning = True
                    except KeyboardInterrupt:
                        durability_warning = True
                        interruption_warning = True
            finally:
                if previous_mask is not None:
                    try:
                        signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)
                    except KeyboardInterrupt:
                        if committed:
                            interruption_warning = True
                        else:
                            raise
        finally:
            if temp_path is not None:
                try:
                    temp_path.unlink()
                except FileNotFoundError:
                    pass
        return ExecutedFileChange(
            canonical_path=target,
            before_hash=self.before_hash,
            after_hash=sha256_bytes(self.proposed_bytes),
            after_mode=mode,
            durability_warning=durability_warning,
            interruption_warning=interruption_warning,
        )

    def _assert_unchanged(self) -> Path:
        target = self._resolver.resolve_write(self.requested_path)
        if target != self.canonical_path:
            raise FileConflictError("path changed since preparation")
        if self.existed_before:
            if not target.exists():
                raise FileConflictError("file changed since preparation")
            try:
                current_stat = target.stat()
            except OSError as error:
                raise FileConflictError("file changed since preparation") from error
            if not stat.S_ISREG(current_stat.st_mode):
                raise FileConflictError("file changed since preparation")
            if current_stat.st_size > self._max_file_bytes:
                raise FileConflictError("file changed since preparation")
            try:
                current = target.read_bytes()
            except OSError as error:
                raise FileConflictError("file changed since preparation") from error
            if sha256_bytes(current) != self.before_hash:
                raise FileConflictError("file changed since preparation")
        elif target.exists():
            raise FileConflictError("file changed since preparation")
        return target


class FileSystemTools:
    """Bounded read/list and two-phase managed text-file writes."""

    def __init__(
        self,
        workspace: str | os.PathLike[str],
        *,
        max_file_bytes: int = DEFAULT_MAX_FILE_BYTES,
        max_output_bytes: int = 64 * 1024,
        max_read_lines: int = DEFAULT_MAX_READ_LINES,
        max_list_entries: int = DEFAULT_MAX_LIST_ENTRIES,
    ) -> None:
        if type(max_file_bytes) is not int or max_file_bytes < 1:
            raise ValueError("max_file_bytes must be a positive integer")
        self.resolver = WorkspaceResolver(workspace)
        self.max_file_bytes = max_file_bytes
        self.max_output_bytes = max_output_bytes
        self.max_read_lines = max_read_lines
        self.max_list_entries = max_list_entries

    def read_file(self, arguments: dict[str, Any]) -> ToolResult:
        path = self.resolver.resolve_read(_string_argument(arguments, "path"))
        text, _ = self._read_regular_text(path)
        offset = _positive_integer(arguments, "offset", default=1)
        limit = _positive_integer(
            arguments, "limit", default=self.max_read_lines, maximum=self.max_read_lines
        )
        lines = text.splitlines()
        if lines and offset > len(lines):
            raise FileToolError("offset is beyond end of file")
        if not lines and offset != 1:
            raise FileToolError("offset is beyond end of file")
        start = offset - 1
        selected = lines[start : start + limit]
        output = "\n".join(
            f"{line_number} | {line}"
            for line_number, line in enumerate(selected, start=offset)
        )
        next_offset = start + len(selected) + 1
        paginated = start + len(selected) < len(lines)
        return ToolResult.bounded(
            title=f"Read {self.resolver.relative_display(path)}",
            output=output,
            metadata={
                "path": str(path),
                "offset": offset,
                "limit": limit,
                "total_lines": len(lines),
                "next_offset": next_offset if paginated else None,
                "truncated": paginated,
            },
            max_bytes=self.max_output_bytes,
            max_lines=limit,
        )

    def list_dir(self, arguments: dict[str, Any]) -> ToolResult:
        requested = arguments.get("path", ".")
        path = self.resolver.resolve_read(_nonempty_string(requested, "path"))
        if not path.is_dir():
            raise FileToolError("path is not a directory")
        limit = _positive_integer(
            arguments,
            "limit",
            default=self.max_list_entries,
            maximum=self.max_list_entries,
        )
        try:
            entries = sorted(path.iterdir(), key=lambda entry: entry.name)
        except OSError as error:
            raise FileToolError(f"cannot list directory: {requested}") from error
        rendered = [self._render_entry(entry) for entry in entries[:limit]]
        paginated = len(entries) > limit
        return ToolResult.bounded(
            title=f"List {self.resolver.relative_display(path)}",
            output="\n".join(rendered),
            metadata={
                "path": str(path),
                "total_entries": len(entries),
                "limit": limit,
                "truncated": paginated,
            },
            max_bytes=self.max_output_bytes,
            max_lines=limit,
        )

    def prepare_write_file(self, arguments: dict[str, Any]) -> PreparedFileChange:
        requested = _string_argument(arguments, "path")
        content = _string_argument(arguments, "content", allow_empty=True)
        proposed = self._encode_text(content)
        path = self.resolver.resolve_write(requested)
        return self._prepare(path, requested, proposed)

    def prepare_edit_file(self, arguments: dict[str, Any]) -> PreparedFileChange:
        requested = _string_argument(arguments, "path")
        old_text = _string_argument(arguments, "old_text", allow_empty=False)
        new_text = _string_argument(arguments, "new_text", allow_empty=True)
        if old_text == new_text:
            raise FileToolError("old_text and new_text must differ")
        path = self.resolver.resolve_write(requested)
        before_text, before_bytes, _ = self._existing_regular_text(path)
        occurrences = before_text.count(old_text)
        if occurrences == 0:
            raise FileToolError("old_text was not found")
        if occurrences > 1:
            raise FileToolError("old_text appears more than once")
        proposed = self._encode_text(before_text.replace(old_text, new_text, 1))
        return self._prepare(
            path, requested, proposed, known_before_bytes=before_bytes
        )

    def _prepare(
        self,
        path: Path,
        requested: str,
        proposed: bytes,
        *,
        known_before_bytes: bytes | None = None,
    ) -> PreparedFileChange:
        parent = path.parent
        if not parent.exists() or not parent.is_dir():
            raise FileToolError(f"parent directory does not exist: {parent}")
        existed = path.exists()
        before_mode: int | None = None
        if existed:
            _, before_bytes, before_mode = self._existing_regular_text(path)
            if known_before_bytes is not None and before_bytes != known_before_bytes:
                raise FileConflictError("file changed during preparation")
        else:
            before_bytes = b""
        before_text = before_bytes.decode("utf-8")
        proposed_text = proposed.decode("utf-8")
        diff = _unified_diff(
            before_text,
            proposed_text,
            from_file=str(path) if existed else "/dev/null",
            to_file=str(path),
        )
        return PreparedFileChange(
            canonical_path=path,
            before_hash=sha256_bytes(before_bytes) if existed else None,
            diff=diff,
            existed_before=existed,
            before_bytes=before_bytes,
            before_mode=before_mode,
            proposed_bytes=proposed,
            requested_path=requested,
            _resolver=self.resolver,
            _max_file_bytes=self.max_file_bytes,
        )

    def _read_regular_text(self, path: Path) -> tuple[str, bytes]:
        text, content, _ = self._existing_regular_text(path)
        return text, content

    def _existing_regular_text(self, path: Path) -> tuple[str, bytes, int]:
        try:
            file_stat = path.stat()
        except (FileNotFoundError, OSError) as error:
            raise FileToolError(f"file does not exist: {path}") from error
        if not stat.S_ISREG(file_stat.st_mode):
            raise FileToolError("path is not a regular file")
        if file_stat.st_size > self.max_file_bytes:
            raise FileToolError("file exceeds size limit")
        try:
            with path.open("rb") as stream:
                content = stream.read(self.max_file_bytes + 1)
        except OSError as error:
            raise FileToolError(f"cannot read file: {path}") from error
        if len(content) > self.max_file_bytes:
            raise FileToolError("file exceeds size limit")
        try:
            text = content.decode("utf-8")
        except UnicodeDecodeError:
            raise FileToolError("file is not valid UTF-8 text") from None
        if "\0" in text:
            raise FileToolError("binary files are not supported")
        return text, content, stat.S_IMODE(file_stat.st_mode)

    def _encode_text(self, content: str) -> bytes:
        if "\0" in content:
            raise FileToolError("binary text is not supported")
        try:
            encoded = content.encode("utf-8")
        except UnicodeEncodeError:
            raise FileToolError("content must be valid UTF-8 text") from None
        if len(encoded) > self.max_file_bytes:
            raise FileToolError("content exceeds size limit")
        return encoded

    @staticmethod
    def _render_entry(entry: Path) -> str:
        if entry.is_symlink():
            suffix = "@"
        elif entry.is_dir():
            suffix = "/"
        elif entry.is_file():
            suffix = ""
        else:
            suffix = "?"
        return entry.name + suffix


def _string_argument(
    arguments: dict[str, Any], key: str, *, allow_empty: bool = False
) -> str:
    if not isinstance(arguments, dict):
        raise FileToolError("arguments must be an object")
    if key not in arguments:
        raise FileToolError(f"missing argument: {key}")
    return _nonempty_string(arguments[key], key, allow_empty=allow_empty)


def _nonempty_string(value: object, name: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or (not allow_empty and not value):
        qualifier = "a string" if allow_empty else "a non-empty string"
        raise FileToolError(f"{name} must be {qualifier}")
    return value


def _positive_integer(
    arguments: dict[str, Any],
    key: str,
    *,
    default: int,
    maximum: int | None = None,
) -> int:
    value = arguments.get(key, default)
    if type(value) is not int or value < 1:
        raise FileToolError(f"{key} must be a positive integer")
    if maximum is not None and value > maximum:
        raise FileToolError(f"{key} must be <= {maximum}")
    return value


def _unified_diff(
    before: str, after: str, *, from_file: str, to_file: str
) -> str:
    if before == after:
        return ""
    lines = difflib.unified_diff(
        before.splitlines(keepends=True),
        after.splitlines(keepends=True),
        fromfile=from_file,
        tofile=to_file,
        lineterm="\n",
    )
    rendered: list[str] = []
    for line in lines:
        if line.endswith("\n"):
            rendered.append(line)
            continue
        rendered.append(line + "\n")
        if line.startswith(("+", "-", " ")) and not line.startswith(
            ("+++", "---")
        ):
            rendered.append("\\ No newline at end of file\n")
    return "".join(rendered)


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
