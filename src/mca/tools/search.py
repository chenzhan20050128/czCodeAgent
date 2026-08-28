"""Bounded workspace content search backed exclusively by ripgrep."""

from __future__ import annotations

import os
import subprocess
import threading
from dataclasses import dataclass
from typing import Any, TextIO

from .filesystem import (
    FileToolError,
    WorkspaceResolver,
    _nonempty_string,
    _optional_path,
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
        max_output_bytes: int = 64 * 1024,
        max_output_lines: int = DEFAULT_MAX_SEARCH_LINES,
    ) -> None:
        if type(max_output_bytes) is not int or max_output_bytes < 1:
            raise ValueError("max_output_bytes must be a positive integer")
        if type(max_output_lines) is not int or max_output_lines < 1:
            raise ValueError("max_output_lines must be a positive integer")
        self.resolver = WorkspaceResolver(workspace)
        self.max_output_bytes = max_output_bytes
        self.max_output_lines = max_output_lines

    def grep(self, arguments: dict[str, Any]) -> ToolResult:
        if not isinstance(arguments, dict):
            raise FileToolError("arguments must be an object")
        pattern = _nonempty_string(arguments.get("pattern"), "pattern")
        requested_path = _optional_path(arguments.get("path"))
        glob = arguments.get("glob")
        if glob is not None:
            glob = _nonempty_string(glob, "glob")
        path = self.resolver.resolve_read(requested_path)
        relative = self.resolver.relative_display(path)
        try:
            collected = self._run_rg(pattern, relative, glob)
        except FileNotFoundError:
            raise FileToolError(
                "ripgrep is required for grep; install it so rg is available"
            ) from None
        return ToolResult.bounded(
            title=f"Search {relative}",
            output=collected.output,
            metadata={
                "path": str(path),
                "pattern": pattern,
                "glob": glob,
                "engine": "rg",
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
