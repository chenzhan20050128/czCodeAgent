"""Bounded workspace content search using rg with a Python fallback."""

from __future__ import annotations

import fnmatch
import os
import re
import subprocess
from pathlib import Path
from typing import Any

from .filesystem import (
    DEFAULT_MAX_FILE_BYTES,
    FileToolError,
    WorkspaceResolver,
    _nonempty_string,
)
from .registry import ToolResult


DEFAULT_MAX_SEARCH_LINES = 500


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
            output = self._run_rg(pattern, relative, glob)
            engine = "rg"
        except FileNotFoundError:
            output = self._python_search(compiled, path, glob)
            engine = "python"
        match_count = len(output.splitlines()) if output else 0
        return ToolResult.bounded(
            title=f"Search {relative}",
            output=output,
            metadata={
                "path": str(path),
                "pattern": pattern,
                "glob": glob,
                "engine": engine,
                "matches": match_count,
            },
            max_bytes=self.max_output_bytes,
            max_lines=self.max_output_lines,
        )

    def _run_rg(self, pattern: str, path: str, glob: str | None) -> str:
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
        completed = subprocess.run(
            argv,
            cwd=self.resolver.workspace,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if completed.returncode not in (0, 1):
            detail = completed.stderr.decode("utf-8", errors="replace").strip()
            suffix = f": {detail}" if detail else ""
            raise FileToolError(
                f"rg failed with exit code {completed.returncode}{suffix}"
            )
        try:
            return completed.stdout.decode("utf-8").rstrip("\n")
        except UnicodeDecodeError:
            raise FileToolError("rg returned invalid UTF-8 output") from None

    def _python_search(
        self, compiled: re.Pattern[str], path: Path, glob: str | None
    ) -> str:
        matches: list[str] = []
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
            relative = file_path.relative_to(self.resolver.workspace).as_posix()
            for line_number, line in enumerate(text.splitlines(), start=1):
                if compiled.search(line):
                    matches.append(f"{relative}:{line_number}:{line}")
        return "\n".join(matches)

    def _candidate_files(self, path: Path, glob: str | None) -> list[Path]:
        if path.is_file():
            candidates = [path] if not path.is_symlink() else []
        elif path.is_dir():
            candidates = []
            for root, directories, names in os.walk(path, followlinks=False):
                directories[:] = sorted(
                    name
                    for name in directories
                    if not name.startswith(".")
                    and not (Path(root) / name).is_symlink()
                )
                for name in sorted(names):
                    candidate = Path(root) / name
                    if name.startswith(".") or candidate.is_symlink():
                        continue
                    candidates.append(candidate)
        else:
            raise FileToolError("search path is not a regular file or directory")
        if glob is None:
            return sorted(candidates, key=lambda candidate: candidate.as_posix())
        return sorted([
            candidate
            for candidate in candidates
            if fnmatch.fnmatch(
                candidate.relative_to(path if path.is_dir() else path.parent).as_posix(),
                glob,
            )
            or fnmatch.fnmatch(candidate.name, glob)
        ], key=lambda candidate: candidate.as_posix())
