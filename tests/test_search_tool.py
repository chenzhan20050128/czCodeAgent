"""Tests for bounded workspace grep backed exclusively by ripgrep."""

from __future__ import annotations

import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mca.tools.filesystem import FileToolError, PathSafetyError
from mca.tools.search import SearchTools


class ChunkedStream:
    def __init__(self, chunks: list[str], on_exhaust: object) -> None:
        self._chunks = list(chunks)
        self._on_exhaust = on_exhaust
        self._exhausted = False
        self.read_calls = 0
        self.bytes_read = 0

    def read(self, size: int = -1) -> str:
        self.read_calls += 1
        if not self._chunks:
            self._mark_exhausted()
            return ""
        chunk = self._chunks[0]
        amount = len(chunk) if size < 0 else min(size, len(chunk))
        output = chunk[:amount]
        self._chunks[0] = chunk[amount:]
        if not self._chunks[0]:
            self._chunks.pop(0)
        self.bytes_read += len(output.encode("utf-8"))
        return output

    def readline(self, size: int = -1) -> str:
        return self.read(size)

    def close(self) -> None:
        self._mark_exhausted()

    def _mark_exhausted(self) -> None:
        if not self._exhausted:
            self._exhausted = True
            self._on_exhaust()


class FakeProcess:
    def __init__(
        self,
        *,
        stdout_chunks: list[str],
        stderr_chunks: list[str] | None = None,
        returncode: int = 0,
    ) -> None:
        self._target_returncode = returncode
        self._exhausted: set[str] = set()
        self._lock = threading.Lock()
        self.returncode: int | None = None
        self.terminated = False
        self.killed = False
        self.stdout = ChunkedStream(
            stdout_chunks, lambda: self._stream_exhausted("stdout")
        )
        self.stderr = ChunkedStream(
            stderr_chunks or [], lambda: self._stream_exhausted("stderr")
        )

    def _stream_exhausted(self, name: str) -> None:
        with self._lock:
            self._exhausted.add(name)
            if self._exhausted == {"stdout", "stderr"} and self.returncode is None:
                self.returncode = self._target_returncode

    def poll(self) -> int | None:
        return self.returncode

    def wait(self, timeout: float | None = None) -> int:
        deadline = None if timeout is None else time.monotonic() + timeout
        while self.returncode is None:
            if deadline is not None and time.monotonic() >= deadline:
                raise subprocess.TimeoutExpired("rg", timeout)
            time.sleep(0.001)
        return self.returncode

    def terminate(self) -> None:
        self.terminated = True
        self.returncode = -15

    def kill(self) -> None:
        self.killed = True
        self.returncode = -9


class SearchToolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.workspace = Path(self.temporary.name) / "work"
        self.workspace.mkdir()
        self.search = SearchTools(
            self.workspace, max_output_bytes=512, max_output_lines=20
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def write_text(self, relative: str, content: str) -> Path:
        path = self.workspace / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path

    def test_rg_uses_argv_without_shell_and_option_terminator(self) -> None:
        process = FakeProcess(
            stdout_chunks=["main.py:1:-needle\n"], returncode=0
        )
        with patch("mca.tools.search.subprocess.Popen", return_value=process) as popen:
            result = self.search.grep(
                {"pattern": "-needle", "path": ".", "glob": "*.py"}
            )

        self.assertEqual(
            popen.call_args.args[0],
            [
                "rg",
                "--line-number",
                "--color",
                "never",
                "--no-heading",
                "--with-filename",
                "--glob",
                "*.py",
                "--",
                "-needle",
                ".",
            ],
        )
        self.assertNotIn("shell", popen.call_args.kwargs)
        self.assertEqual(popen.call_args.kwargs["cwd"], self.workspace.resolve())
        self.assertEqual(result.output, "main.py:1:-needle")
        self.assertIs(result.metadata["truncated"], False)
        self.assertEqual(result.metadata["matches_seen"], 1)
        self.assertEqual(result.metadata["matches_stored"], 1)

    def test_no_matches_is_a_successful_empty_result(self) -> None:
        process = FakeProcess(stdout_chunks=[], returncode=1)
        with patch("mca.tools.search.subprocess.Popen", return_value=process):
            result = self.search.grep({"pattern": "missing"})

        self.assertEqual(result.output, "")
        self.assertEqual(result.status, "succeeded")
        self.assertEqual(result.metadata["matches"], 0)

    def test_rg_failure_is_stable_tool_error(self) -> None:
        process = FakeProcess(
            stdout_chunks=[], stderr_chunks=["permission denied\n"], returncode=2
        )
        with patch("mca.tools.search.subprocess.Popen", return_value=process):
            with self.assertRaisesRegex(FileToolError, "rg failed with exit code 2"):
                self.search.grep({"pattern": "needle"})

    def test_rg_error_stderr_detail_is_memory_bounded(self) -> None:
        process = FakeProcess(
            stdout_chunks=[],
            stderr_chunks=["x" * 100 for _ in range(100)],
            returncode=2,
        )
        with patch("mca.tools.search.subprocess.Popen", return_value=process):
            with self.assertRaises(FileToolError) as raised:
                self.search.grep({"pattern": "needle"})

        self.assertLessEqual(len(str(raised.exception).encode()), 600)

    def test_invalid_regex_is_reported_from_bounded_rg_stderr(self) -> None:
        process = FakeProcess(
            stdout_chunks=[],
            stderr_chunks=["regex parse error: unclosed character class\n"],
            returncode=2,
        )
        with patch("mca.tools.search.subprocess.Popen", return_value=process) as popen:
            with self.assertRaisesRegex(FileToolError, "regex parse error"):
                self.search.grep({"pattern": "["})

        popen.assert_called_once()

    def test_missing_ripgrep_returns_stable_installation_error(self) -> None:
        with patch(
            "mca.tools.search.subprocess.Popen", side_effect=FileNotFoundError
        ):
            with self.assertRaisesRegex(
                FileToolError,
                "ripgrep.*required.*install.*rg",
            ):
                self.search.grep({"pattern": "needle"})

    def test_result_is_bounded_by_lines_and_utf8_bytes(self) -> None:
        bounded = SearchTools(
            self.workspace, max_output_bytes=55, max_output_lines=3
        )
        process = FakeProcess(
            stdout_chunks=[
                line
                for line in (
                    "a:1:甲乙丙\n",
                    "a:2:甲乙丙\n",
                    "a:3:甲乙丙\n",
                    "a:4:甲乙丙\n",
                    "a:5:must not be consumed\n",
                )
            ],
        )
        with patch("mca.tools.search.subprocess.Popen", return_value=process):
            result = bounded.grep({"pattern": "甲"})

        self.assertIs(result.metadata["truncated"], True)
        self.assertLessEqual(len(result.output.splitlines()), 3)
        self.assertLessEqual(len(result.output.encode("utf-8")), 55)
        self.assertIn("truncated", result.output)
        self.assertTrue(process.terminated)
        self.assertLess(process.stdout.read_calls, 6)
        self.assertGreaterEqual(
            result.metadata["matches_seen"], result.metadata["matches_stored"]
        )

    def test_rg_stops_reading_an_arbitrarily_long_stream_at_the_preview_cap(self) -> None:
        bounded = SearchTools(
            self.workspace, max_output_bytes=128, max_output_lines=3
        )
        process = FakeProcess(
            stdout_chunks=[f"file:{index}:needle\n" for index in range(10_000)]
        )

        with patch("mca.tools.search.subprocess.Popen", return_value=process):
            result = bounded.grep({"pattern": "needle"})

        self.assertTrue(process.terminated)
        self.assertLessEqual(process.stdout.read_calls, 5)
        self.assertLess(process.stdout.bytes_read, 128)
        self.assertIs(result.metadata["truncated"], True)
        self.assertIs(result.metadata["matches_complete"], False)

    def test_search_module_has_no_python_regex_or_filesystem_fallback(self) -> None:
        import mca.tools.search as search_module

        source = Path(search_module.__file__).read_text(encoding="utf-8")
        self.assertNotIn("import re", source)
        self.assertNotIn("re.compile", source)
        self.assertNotIn("os.walk", source)
        self.assertNotIn("_python_search", source)

    def test_path_outside_workspace_is_rejected(self) -> None:
        for path in ("../outside", "/etc"):
            with self.subTest(path=path):
                with self.assertRaises(PathSafetyError):
                    self.search.grep({"pattern": "x", "path": path})

    def test_grep_accepts_a_workspace_internal_absolute_path(self) -> None:
        self.write_text("pkg/mod.py", "needle here\n")
        process = FakeProcess(stdout_chunks=["pkg/mod.py:1:needle here\n"])
        with patch("mca.tools.search.subprocess.Popen", return_value=process):
            result = self.search.grep(
                {"pattern": "needle", "path": str(self.workspace / "pkg")}
            )
        self.assertEqual(result.status, "succeeded")
        self.assertIn("needle", result.output)


if __name__ == "__main__":
    unittest.main()
