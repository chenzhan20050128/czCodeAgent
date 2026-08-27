"""Bounded workspace content search using rg with a Python fallback."""

from __future__ import annotations

import fnmatch
import os
import re
import subprocess
import threading
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO

from .filesystem import (
    DEFAULT_MAX_FILE_BYTES,
    FileToolError,
    WorkspaceResolver,
    _nonempty_string,
)
from .registry import TRUNCATION_MARKER, ToolResult, truncate_output


DEFAULT_MAX_SEARCH_LINES = 500
_MAX_ERROR_BYTES = 512
_TERMINATE_TIMEOUT_SECONDS = 1.0


@dataclass(frozen=True)
class _SearchOutput:
    output: str
    matches_seen: int
    matches_stored: int
    truncated: bool
    matches_complete: bool


class _PreviewCollector:
    """Keep at most one bounded search preview plus one bounded candidate."""

    def __init__(self, *, max_bytes: int, max_lines: int) -> None:
        self.max_bytes = max_bytes
        self.max_lines = max_lines
        self.output = ""
        self.matches_seen = 0
        self.matches_stored = 0
        self.truncated = False

    def add(self, line: str) -> bool:
        """Add one match; return false once the bounded preview is full."""

        self.matches_seen += 1
        bounded_line = _utf8_prefix(line, self.max_bytes + 1)
        candidate = bounded_line if not self.output else f"{self.output}\n{bounded_line}"
        if (
            len(candidate.encode("utf-8")) > self.max_bytes
            or len(candidate.splitlines()) > self.max_lines
            or bounded_line != line
        ):
            self.output, _ = truncate_output(
                candidate, max_bytes=self.max_bytes, max_lines=self.max_lines
            )
            self.truncated = True
            self.matches_stored = sum(
                rendered != TRUNCATION_MARKER
                for rendered in self.output.splitlines()
            )
            return False
        self.output = candidate
        self.matches_stored = self.matches_seen
        return True

    def finish(self, *, complete: bool) -> _SearchOutput:
        return _SearchOutput(
            output=self.output,
            matches_seen=self.matches_seen,
            matches_stored=self.matches_stored,
            truncated=self.truncated,
            matches_complete=complete and not self.truncated,
        )


class _BoundedTextBuffer:
    """Drain a text stream while retaining only a byte-bounded prefix."""

    def __init__(self, max_bytes: int) -> None:
        self.max_bytes = max_bytes
        self.value = ""

    def append(self, chunk: str) -> None:
        remaining = self.max_bytes - len(self.value.encode("utf-8"))
        if remaining > 0:
            self.value += _utf8_prefix(chunk, remaining)


class SearchTools:
    """Search text below one workspace without invoking a shell."""

    def __init__(
        self,
        workspace: str | os.PathLike[str],
        *,
        max_file_bytes: int = DEFAULT_MAX_FILE_BYTES,
        max_output_bytes: int = 64 * 1024,
        max_output_lines: int = DEFAULT_MAX_SEARCH_LINES,
    ) -> None:
        if type(max_output_bytes) is not int or max_output_bytes < 1:
            raise ValueError("max_output_bytes must be a positive integer")
        if type(max_output_lines) is not int or max_output_lines < 1:
            raise ValueError("max_output_lines must be a positive integer")
        self.resolver = WorkspaceResolver(workspace)
        self.max_file_bytes = max_file_bytes
        self.max_output_bytes = max_output_bytes
        self.max_output_lines = max_output_lines

    def grep(self, arguments: dict[str, Any]) -> ToolResult:
        if not isinstance(arguments, dict):
            raise FileToolError("arguments must be an object")
        pattern = _nonempty_string(arguments.get("pattern"), "pattern")
        requested_path = _nonempty_string(arguments.get("path", "."), "path")
        glob = arguments.get("glob")
        if glob is not None:
            glob = _nonempty_string(glob, "glob")
        try:
            compiled = re.compile(pattern)
        except re.error as error:
            raise FileToolError(f"invalid regular expression: {error}") from None
        path = self.resolver.resolve_read(requested_path)
        relative = self.resolver.relative_display(path)
        try:
            collected = self._run_rg(pattern, relative, glob)
            engine = "rg"
        except FileNotFoundError:
            collected = self._python_search(compiled, path, glob)
            engine = "python"
        return ToolResult.bounded(
            title=f"Search {relative}",
            output=collected.output,
            metadata={
                "path": str(path),
                "pattern": pattern,
                "glob": glob,
                "engine": engine,
                "matches": collected.matches_stored,
                "matches_seen": collected.matches_seen,
                "matches_stored": collected.matches_stored,
                "matches_complete": collected.matches_complete,
                "truncated": collected.truncated,
            },
            max_bytes=self.max_output_bytes,
            max_lines=self.max_output_lines,
        )

    def _run_rg(
        self, pattern: str, path: str, glob: str | None
    ) -> _SearchOutput:
        argv = [
            "rg",
            "--line-number",
            "--color",
            "never",
            "--no-heading",
            "--with-filename",
        ]
        if glob is not None:
            argv.extend(("--glob", glob))
        argv.extend(("--", pattern, path))
        process = subprocess.Popen(
            argv,
            cwd=self.resolver.workspace,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        if process.stdout is None or process.stderr is None:
            process.kill()
            process.wait()
            raise FileToolError("rg pipes were not created")

        stderr = _BoundedTextBuffer(_MAX_ERROR_BYTES)
        stderr_thread = threading.Thread(
            target=_drain_text_stream, args=(process.stderr, stderr), daemon=True
        )
        stderr_thread.start()
        collector = _PreviewCollector(
            max_bytes=self.max_output_bytes, max_lines=self.max_output_lines
        )
        stopped_early = False
        try:
            while True:
                line = process.stdout.readline(self.max_output_bytes + 1)
                if line == "":
                    break
                if not collector.add(line.rstrip("\r\n")):
                    stopped_early = True
                    _terminate_process(process)
                    break
            if not stopped_early:
                process.wait()
        finally:
            process.stdout.close()
            if process.poll() is None:
                _terminate_process(process)
            process.stderr.close()
            stderr_thread.join(timeout=_TERMINATE_TIMEOUT_SECONDS)

        return_code = process.returncode
        if not stopped_early and return_code not in (0, 1):
            detail = stderr.value.strip()
            suffix = f": {detail}" if detail else ""
            raise FileToolError(f"rg failed with exit code {return_code}{suffix}")
        return collector.finish(complete=not stopped_early)

    def _python_search(
        self, compiled: re.Pattern[str], path: Path, glob: str | None
    ) -> _SearchOutput:
        collector = _PreviewCollector(
            max_bytes=self.max_output_bytes, max_lines=self.max_output_lines
        )
        for file_path in self._candidate_files(path, glob):
            try:
                file_stat = file_path.stat()
                if file_stat.st_size > self.max_file_bytes:
                    continue
                content = file_path.read_bytes()
            except OSError:
                continue
            if len(content) > self.max_file_bytes or b"\0" in content:
                continue
            try:
                text = content.decode("utf-8")
            except UnicodeDecodeError:
                continue
            relative = self.resolver.relative_display(file_path.resolve())
            for line_number, line in enumerate(text.splitlines(), start=1):
                if compiled.search(line) and not collector.add(
                    f"{relative}:{line_number}:{line}"
                ):
                    return collector.finish(complete=False)
        return collector.finish(complete=True)

    def _candidate_files(
        self, path: Path, glob: str | None
    ) -> Iterator[Path]:
        if path.is_file():
            if not path.is_symlink() and _matches_glob(path, path.parent, glob):
                yield path
            return
        if not path.is_dir():
            raise FileToolError("search path is not a regular file or directory")
        yield from self._walk_directory(path, path, glob)

    def _walk_directory(
        self, directory: Path, search_root: Path, glob: str | None
    ) -> Iterator[Path]:
        try:
            entries = sorted(directory.iterdir(), key=lambda entry: entry.name)
        except OSError:
            return
        for entry in entries:
            if entry.name.startswith(".") or entry.is_symlink():
                continue
            if entry.is_dir():
                yield from self._walk_directory(entry, search_root, glob)
            elif entry.is_file() and _matches_glob(entry, search_root, glob):
                yield entry


def _matches_glob(candidate: Path, root: Path, glob: str | None) -> bool:
    if glob is None:
        return True
    relative = candidate.relative_to(root).as_posix()
    return fnmatch.fnmatch(relative, glob) or fnmatch.fnmatch(candidate.name, glob)


def _drain_text_stream(stream: TextIO, destination: _BoundedTextBuffer) -> None:
    while True:
        chunk = stream.read(4096)
        if chunk == "":
            return
        destination.append(chunk)


def _terminate_process(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=_TERMINATE_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def _utf8_prefix(value: str, max_bytes: int) -> str:
    encoded = value.encode("utf-8")
    clipped = encoded[:max_bytes]
    while clipped:
        try:
            return clipped.decode("utf-8")
        except UnicodeDecodeError:
            clipped = clipped[:-1]
    return ""
