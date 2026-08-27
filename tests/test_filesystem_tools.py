"""Tests for workspace-confined text-file tools."""

from __future__ import annotations

import hashlib
import os
import stat
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mca.tools.filesystem import (
    FileConflictError,
    FileSystemTools,
    FileToolError,
    PathSafetyError,
    WorkspaceResolver,
)


class FileSystemToolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.workspace = Path(self.temporary.name) / "work"
        self.workspace.mkdir()
        self.tools = FileSystemTools(
            self.workspace, max_file_bytes=1_024, max_output_bytes=1_024
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def write_bytes(self, relative: str, content: bytes) -> Path:
        path = self.workspace / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        return path

    def test_resolver_accepts_relative_path_inside_workspace(self) -> None:
        path = self.write_bytes("src/main.py", b"pass\n")
        resolver = WorkspaceResolver(self.workspace)

        self.assertEqual(resolver.resolve_read("src/main.py"), path.resolve())

    def test_resolver_rejects_empty_absolute_nul_and_escape_paths(self) -> None:
        resolver = WorkspaceResolver(self.workspace)
        sibling = self.workspace.with_name(self.workspace.name + "-sibling")
        sibling.mkdir()
        cases = [
            "",
            str(self.workspace / "file.txt"),
            "bad\0name",
            "../outside.txt",
            f"../{sibling.name}/file.txt",
        ]
        for value in cases:
            with self.subTest(value=value):
                with self.assertRaises(PathSafetyError):
                    resolver.resolve_read(value, must_exist=False)

    def test_read_can_follow_an_internal_symlink_but_not_an_external_one(self) -> None:
        target = self.write_bytes("real.txt", b"inside")
        (self.workspace / "internal.txt").symlink_to(target)
        outside = self.workspace.parent / "outside.txt"
        outside.write_text("outside", encoding="utf-8")
        (self.workspace / "external.txt").symlink_to(outside)

        self.assertEqual(
            self.tools.resolver.resolve_read("internal.txt"), target.resolve()
        )
        with self.assertRaises(PathSafetyError):
            self.tools.resolver.resolve_read("external.txt")

    def test_write_rejects_a_symlink_target_or_parent_even_when_final_is_missing(self) -> None:
        real_dir = self.workspace / "real"
        real_dir.mkdir()
        (self.workspace / "linked-dir").symlink_to(real_dir, target_is_directory=True)
        target = self.write_bytes("target.txt", b"old")
        (self.workspace / "linked-file.txt").symlink_to(target)

        for value in ("linked-file.txt", "linked-dir/new.txt"):
            with self.subTest(value=value):
                with self.assertRaisesRegex(PathSafetyError, "symbolic link"):
                    self.tools.prepare_write_file({"path": value, "content": "new"})

    def test_read_file_uses_one_based_pagination_and_line_numbers(self) -> None:
        self.write_bytes("notes.txt", b"alpha\nbeta\ngamma\ndelta\n")

        result = self.tools.read_file({"path": "notes.txt", "offset": 2, "limit": 2})

        self.assertEqual(result.output, "2 | beta\n3 | gamma")
        self.assertEqual(result.metadata["offset"], 2)
        self.assertEqual(result.metadata["next_offset"], 4)
        self.assertIs(result.metadata["truncated"], True)

    def test_read_file_rejects_invalid_bounds(self) -> None:
        self.write_bytes("one.txt", b"one\n")
        for arguments in (
            {"path": "one.txt", "offset": 0},
            {"path": "one.txt", "offset": 2},
            {"path": "one.txt", "limit": 0},
            {"path": "one.txt", "offset": True},
        ):
            with self.subTest(arguments=arguments):
                with self.assertRaises(FileToolError):
                    self.tools.read_file(arguments)

    def test_read_file_rejects_large_binary_invalid_utf8_and_non_regular_targets(self) -> None:
        self.write_bytes("large.txt", b"x" * 1_025)
        self.write_bytes("binary.txt", b"hello\0world")
        self.write_bytes("invalid.txt", b"\xff")
        (self.workspace / "folder").mkdir()
        for value in ("large.txt", "binary.txt", "invalid.txt", "folder"):
            with self.subTest(value=value):
                with self.assertRaises(FileToolError):
                    self.tools.read_file({"path": value})

    def test_empty_file_can_be_read_at_first_line(self) -> None:
        self.write_bytes("empty.txt", b"")
        result = self.tools.read_file({"path": "empty.txt"})
        self.assertEqual(result.output, "")
        self.assertEqual(result.metadata["total_lines"], 0)

    def test_list_dir_is_sorted_and_marks_entry_types(self) -> None:
        self.write_bytes("z.txt", b"z")
        self.write_bytes("a.txt", b"a")
        (self.workspace / "folder").mkdir()
        (self.workspace / "link").symlink_to(self.workspace / "a.txt")

        result = self.tools.list_dir({"path": "."})

        self.assertEqual(result.output.splitlines(), ["a.txt", "folder/", "link@", "z.txt"])

    def test_list_dir_is_bounded_and_reports_truncation(self) -> None:
        for name in ("c", "a", "b"):
            self.write_bytes(name, name.encode())

        result = self.tools.list_dir({"path": ".", "limit": 2})

        self.assertEqual(result.output.splitlines(), ["a", "b"])
        self.assertIs(result.metadata["truncated"], True)
        self.assertEqual(result.metadata["total_entries"], 3)

    def test_prepare_write_captures_snapshot_hash_and_diff_without_writing(self) -> None:
        path = self.write_bytes("file.txt", b"old\n")
        path.chmod(0o640)

        change = self.tools.prepare_write_file(
            {"path": "file.txt", "content": "new\n"}
        )

        self.assertEqual(path.read_bytes(), b"old\n")
        self.assertEqual(change.canonical_path, path.resolve())
        self.assertEqual(change.before_hash, hashlib.sha256(b"old\n").hexdigest())
        self.assertTrue(change.existed_before)
        self.assertEqual(change.before_bytes, b"old\n")
        self.assertEqual(change.before_mode, 0o640)
        self.assertEqual(change.proposed_bytes, b"new\n")
        self.assertIn("-old", change.diff)
        self.assertIn("+new", change.diff)

    def test_prepare_new_file_records_absent_snapshot_without_creating_it(self) -> None:
        change = self.tools.prepare_write_file(
            {"path": "new.txt", "content": "hello\n"}
        )

        self.assertFalse((self.workspace / "new.txt").exists())
        self.assertFalse(change.existed_before)
        self.assertIsNone(change.before_hash)
        self.assertIsNone(change.before_mode)
        self.assertIn("--- /dev/null", change.diff)

    def test_prepare_diff_exposes_removed_trailing_newline(self) -> None:
        self.write_bytes("newline.txt", b"same\n")

        change = self.tools.prepare_write_file(
            {"path": "newline.txt", "content": "same"}
        )

        self.assertNotEqual(change.diff, "")
        self.assertIn("+same\n\\ No newline at end of file\n", change.diff)

    def test_prepare_diff_exposes_added_trailing_newline(self) -> None:
        self.write_bytes("newline.txt", b"same")

        change = self.tools.prepare_write_file(
            {"path": "newline.txt", "content": "same\n"}
        )

        self.assertIn("-same\n\\ No newline at end of file\n", change.diff)
        self.assertTrue(change.diff.endswith("+same\n"))

    def test_prepare_diff_renders_ordinary_line_replacement_without_marker(self) -> None:
        self.write_bytes("ordinary.txt", b"old\n")

        change = self.tools.prepare_write_file(
            {"path": "ordinary.txt", "content": "new\n"}
        )

        self.assertIn("-old\n+new\n", change.diff)
        self.assertNotIn("No newline at end of file", change.diff)

    def test_prepare_edit_replaces_exactly_one_match_without_writing(self) -> None:
        path = self.write_bytes("edit.txt", b"before old after\n")

        change = self.tools.prepare_edit_file(
            {"path": "edit.txt", "old_text": "old", "new_text": "new"}
        )

        self.assertEqual(path.read_text(encoding="utf-8"), "before old after\n")
        self.assertEqual(change.proposed_bytes, b"before new after\n")

    def test_prepare_edit_rejects_zero_multiple_or_identical_replacements(self) -> None:
        self.write_bytes("edit.txt", b"same same")
        cases = [
            ("missing", "new", "not found"),
            ("same", "new", "more than once"),
            ("same same", "same same", "must differ"),
        ]
        for old_text, new_text, message in cases:
            with self.subTest(old_text=old_text):
                with self.assertRaisesRegex(FileToolError, message):
                    self.tools.prepare_edit_file(
                        {"path": "edit.txt", "old_text": old_text, "new_text": new_text}
                    )

    def test_prepare_write_rejects_nul_and_oversized_text(self) -> None:
        for content in ("binary\0text", "x" * 1_025):
            with self.subTest(length=len(content)):
                with self.assertRaises(FileToolError):
                    self.tools.prepare_write_file({"path": "file.txt", "content": content})

    def test_execute_uses_same_directory_temp_atomic_replace_and_preserves_mode(self) -> None:
        path = self.write_bytes("file.txt", b"old")
        path.chmod(0o640)
        change = self.tools.prepare_write_file({"path": "file.txt", "content": "new"})
        real_replace = os.replace
        replacements: list[tuple[Path, Path]] = []

        def recording_replace(source: str | os.PathLike[str], target: str | os.PathLike[str]) -> None:
            replacements.append((Path(source), Path(target)))
            real_replace(source, target)

        with patch("mca.tools.filesystem.os.replace", side_effect=recording_replace):
            result = change.execute()

        self.assertEqual(path.read_bytes(), b"new")
        self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o640)
        self.assertEqual(replacements[0][0].parent.resolve(), path.parent.resolve())
        self.assertEqual(replacements[0][1], path.resolve())
        self.assertEqual(result.after_hash, hashlib.sha256(b"new").hexdigest())

    def test_execute_new_file_uses_mode_0644(self) -> None:
        change = self.tools.prepare_write_file({"path": "new.txt", "content": "new"})
        change.execute()
        self.assertEqual(stat.S_IMODE((self.workspace / "new.txt").stat().st_mode), 0o644)

    def test_execute_fsyncs_file_and_parent_directory(self) -> None:
        change = self.tools.prepare_write_file({"path": "new.txt", "content": "new"})
        real_fsync = os.fsync
        calls: list[int] = []

        def recording_fsync(fd: int) -> None:
            calls.append(fd)
            real_fsync(fd)

        with patch("mca.tools.filesystem.os.fsync", side_effect=recording_fsync):
            change.execute()

        self.assertEqual(len(calls), 2)

    def test_execute_rejects_approval_time_hash_conflict(self) -> None:
        path = self.write_bytes("file.txt", b"old")
        change = self.tools.prepare_write_file({"path": "file.txt", "content": "approved"})
        path.write_bytes(b"external")

        with self.assertRaisesRegex(FileConflictError, "changed since preparation"):
            change.execute()

        self.assertEqual(path.read_bytes(), b"external")

    def test_execute_rechecks_symlinks_after_preparation(self) -> None:
        real_dir = self.workspace / "dir"
        real_dir.mkdir()
        change = self.tools.prepare_write_file({"path": "dir/new.txt", "content": "approved"})
        real_dir.rmdir()
        outside = self.workspace.parent / "outside-dir"
        outside.mkdir()
        real_dir.symlink_to(outside, target_is_directory=True)

        with self.assertRaises(PathSafetyError):
            change.execute()
        self.assertFalse((outside / "new.txt").exists())


if __name__ == "__main__":
    unittest.main()
