"""Tests for bounded workspace grep with an rg-first fallback."""

from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mca.tools.filesystem import FileToolError, PathSafetyError
from mca.tools.search import SearchTools


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
        completed = subprocess.CompletedProcess(
            args=[], returncode=0, stdout=b"main.py:1:-needle\n", stderr=b""
        )
        with patch("mca.tools.search.subprocess.run", return_value=completed) as run:
            result = self.search.grep(
                {"pattern": "-needle", "path": ".", "glob": "*.py"}
            )

        self.assertEqual(
            run.call_args.args[0],
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
        self.assertNotIn("shell", run.call_args.kwargs)
        self.assertEqual(run.call_args.kwargs["cwd"], self.workspace.resolve())
        self.assertEqual(result.output, "main.py:1:-needle")

    def test_no_matches_is_a_successful_empty_result(self) -> None:
        completed = subprocess.CompletedProcess(
            args=[], returncode=1, stdout=b"", stderr=b""
        )
        with patch("mca.tools.search.subprocess.run", return_value=completed):
            result = self.search.grep({"pattern": "missing"})

        self.assertEqual(result.output, "")
        self.assertEqual(result.status, "succeeded")
        self.assertEqual(result.metadata["matches"], 0)

    def test_rg_failure_is_stable_tool_error(self) -> None:
        completed = subprocess.CompletedProcess(
            args=[], returncode=2, stdout=b"", stderr=b"permission denied\n"
        )
        with patch("mca.tools.search.subprocess.run", return_value=completed):
            with self.assertRaisesRegex(FileToolError, "rg failed with exit code 2"):
                self.search.grep({"pattern": "needle"})

    def test_invalid_regex_is_rejected_before_running_rg(self) -> None:
        with patch("mca.tools.search.subprocess.run") as run:
            with self.assertRaisesRegex(FileToolError, "invalid regular expression"):
                self.search.grep({"pattern": "["})
        run.assert_not_called()

    def test_fallback_searches_real_files_in_deterministic_order(self) -> None:
        self.write_text("z.txt", "none\nneedle z\n")
        self.write_text("a.txt", "needle a\n")
        self.write_text("sub/b.txt", "first\nneedle b\n")

        with patch("mca.tools.search.subprocess.run", side_effect=FileNotFoundError):
            result = self.search.grep({"pattern": "needle", "path": "."})

        self.assertEqual(
            result.output.splitlines(),
            ["a.txt:1:needle a", "sub/b.txt:2:needle b", "z.txt:2:needle z"],
        )
        self.assertEqual(result.metadata["engine"], "python")
        self.assertEqual(result.metadata["matches"], 3)

    def test_fallback_honors_glob_and_skips_hidden_symlink_binary_and_invalid_utf8(self) -> None:
        self.write_text("keep.py", "needle")
        self.write_text("skip.txt", "needle")
        self.write_text(".hidden.py", "needle")
        hidden = self.workspace / ".hidden-dir"
        hidden.mkdir()
        (hidden / "inside.py").write_text("needle", encoding="utf-8")
        (self.workspace / "binary.py").write_bytes(b"needle\0data")
        (self.workspace / "invalid.py").write_bytes(b"needle\xff")
        (self.workspace / "linked.py").symlink_to(self.workspace / "keep.py")

        with patch("mca.tools.search.subprocess.run", side_effect=FileNotFoundError):
            result = self.search.grep(
                {"pattern": "needle", "path": ".", "glob": "*.py"}
            )

        self.assertEqual(result.output, "keep.py:1:needle")

    def test_result_is_bounded_by_lines_and_utf8_bytes(self) -> None:
        bounded = SearchTools(
            self.workspace, max_output_bytes=55, max_output_lines=3
        )
        completed = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout=("a:1:甲乙丙\na:2:甲乙丙\na:3:甲乙丙\na:4:甲乙丙\n").encode(),
            stderr=b"",
        )
        with patch("mca.tools.search.subprocess.run", return_value=completed):
            result = bounded.grep({"pattern": "甲"})

        self.assertIs(result.metadata["truncated"], True)
        self.assertLessEqual(len(result.output.splitlines()), 3)
        self.assertLessEqual(len(result.output.encode("utf-8")), 55)
        self.assertIn("truncated", result.output)

    def test_path_must_remain_in_workspace(self) -> None:
        for path in ("../outside", str(self.workspace)):
            with self.subTest(path=path):
                with self.assertRaises(PathSafetyError):
                    self.search.grep({"pattern": "x", "path": path})


if __name__ == "__main__":
    unittest.main()
