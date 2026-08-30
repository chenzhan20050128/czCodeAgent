"""Tests for workspace-confined text-file tools."""

from __future__ import annotations

import hashlib
import os
import signal
import stat
import sys
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError
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
from mca.file_versions import DirectoryMutationLease
from mca.file_versions import FileMutationCoordinator


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

    def test_resolver_accepts_a_workspace_internal_absolute_path(self) -> None:
        path = self.write_bytes("src/main.py", b"pass\n")
        resolver = WorkspaceResolver(self.workspace)

        absolute = str(self.workspace / "src" / "main.py")
        self.assertEqual(resolver.resolve_read(absolute), path.resolve())

    def test_resolver_accepts_the_workspace_root_as_an_absolute_path(self) -> None:
        resolver = WorkspaceResolver(self.workspace)

        self.assertEqual(
            resolver.resolve_read(str(self.workspace)), self.workspace.resolve()
        )

    def test_resolver_rejects_empty_nul_and_escaping_paths(self) -> None:
        resolver = WorkspaceResolver(self.workspace)
        sibling = self.workspace.with_name(self.workspace.name + "-sibling")
        sibling.mkdir()
        cases = [
            "",
            str(sibling / "file.txt"),
            "/etc/hosts",
            "C:\\Windows\\system32",
            "bad\0name",
            "../outside.txt",
            f"../{sibling.name}/file.txt",
        ]
        for value in cases:
            with self.subTest(value=value):
                with self.assertRaises(PathSafetyError):
                    resolver.resolve_read(value, must_exist=False)

    def test_write_accepts_a_workspace_internal_absolute_path(self) -> None:
        change = self.tools.prepare_write_file(
            {"path": str(self.workspace / "out.txt"), "content": "hi\n"}
        )
        change.execute()

        self.assertEqual((self.workspace / "out.txt").read_text(encoding="utf-8"), "hi\n")

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

    def test_list_dir_defaults_missing_or_blank_path_to_workspace_root(self) -> None:
        self.write_bytes("a.txt", b"a")

        for arguments in ({}, {"path": ""}, {"path": "   "}):
            with self.subTest(arguments=arguments):
                result = self.tools.list_dir(dict(arguments))
                self.assertEqual(result.output.splitlines(), ["a.txt"])
                self.assertEqual(result.metadata["path"], str(self.workspace.resolve()))

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
        self.assertTrue(change.expected_version.exists)
        self.assertEqual(change.expected_version.sha256, change.before_hash)
        self.assertEqual(change.expected_version.mode, 0o640)
        self.assertEqual(change.expected_version.size, len(b"old\n"))
        self.assertIsNotNone(change.expected_version.device)
        self.assertIsNotNone(change.expected_version.inode)
        self.assertIsNotNone(change.expected_version.mtime_ns)
        self.assertIsNotNone(change.expected_version.ctime_ns)
        with self.assertRaises(FrozenInstanceError):
            change.expected_version.mode = 0o600  # type: ignore[misc]

    def test_prepare_new_file_records_absent_snapshot_without_creating_it(self) -> None:
        change = self.tools.prepare_write_file(
            {"path": "new.txt", "content": "hello\n"}
        )

        self.assertFalse((self.workspace / "new.txt").exists())
        self.assertFalse(change.existed_before)
        self.assertIsNone(change.before_hash)
        self.assertIsNone(change.before_mode)
        self.assertFalse(change.expected_version.exists)
        self.assertIsNone(change.expected_version.sha256)
        self.assertIsNone(change.expected_version.mode)
        self.assertIsNone(change.expected_version.size)
        self.assertIn("--- /dev/null", change.diff)

    def test_mutation_coordinator_releases_idle_path_queues(self) -> None:
        coordinator = FileMutationCoordinator()
        path = (self.workspace / "file.txt").resolve()

        with coordinator.mutation(path, ()):
            self.assertEqual(len(coordinator._paths), 1)

        self.assertEqual(coordinator._paths, {})

    def test_different_file_commits_can_overlap(self) -> None:
        first = self.write_bytes("first.txt", b"old")
        second = self.write_bytes("second.txt", b"old")
        changes = (
            self.tools.prepare_write_file(
                {"path": "first.txt", "content": "first"}
            ),
            self.tools.prepare_write_file(
                {"path": "second.txt", "content": "second"}
            ),
        )
        barrier = threading.Barrier(2)
        real_replace = os.replace

        def overlapping_replace(source: object, target: object) -> None:
            barrier.wait(timeout=2)
            real_replace(source, target)

        with patch(
            "mca.tools.filesystem.os.replace", side_effect=overlapping_replace
        ):
            with ThreadPoolExecutor(max_workers=2) as pool:
                results = list(pool.map(lambda change: change.execute(), changes))

        self.assertEqual(first.read_text(encoding="utf-8"), "first")
        self.assertEqual(second.read_text(encoding="utf-8"), "second")
        self.assertEqual(len(results), 2)

    def test_same_version_same_file_writes_have_one_stale_loser(self) -> None:
        path = self.write_bytes("shared.txt", b"old")
        changes = (
            self.tools.prepare_write_file(
                {"path": "shared.txt", "content": "first"}
            ),
            self.tools.prepare_write_file(
                {"path": "shared.txt", "content": "second"}
            ),
        )

        def commit(change):
            try:
                return ("succeeded", change.execute())
            except FileConflictError as error:
                return ("conflict", str(error))

        with ThreadPoolExecutor(max_workers=2) as pool:
            outcomes = list(pool.map(commit, changes))

        self.assertEqual(
            sorted(status for status, _ in outcomes), ["conflict", "succeeded"]
        )
        conflict = next(detail for status, detail in outcomes if status == "conflict")
        self.assertIn("FILE_STALE_VERSION", conflict)
        self.assertIn(path.read_text(encoding="utf-8"), {"first", "second"})

    def test_same_version_write_and_edit_have_one_stale_loser(self) -> None:
        path = self.write_bytes("shared.txt", b"old text")
        changes = (
            self.tools.prepare_write_file(
                {"path": "shared.txt", "content": "whole write"}
            ),
            self.tools.prepare_edit_file(
                {
                    "path": "shared.txt",
                    "old_text": "old",
                    "new_text": "edited",
                }
            ),
        )

        def commit(change):
            try:
                change.execute()
                return "succeeded"
            except FileConflictError as error:
                self.assertIn("FILE_STALE_VERSION", str(error))
                return "conflict"

        with ThreadPoolExecutor(max_workers=2) as pool:
            outcomes = list(pool.map(commit, changes))

        self.assertEqual(sorted(outcomes), ["conflict", "succeeded"])
        self.assertIn(
            path.read_text(encoding="utf-8"), {"whole write", "edited text"}
        )

    def test_same_absent_file_creation_has_one_stale_loser(self) -> None:
        changes = (
            self.tools.prepare_write_file(
                {"path": "new.txt", "content": "first"}
            ),
            self.tools.prepare_write_file(
                {"path": "new.txt", "content": "second"}
            ),
        )

        def commit(change):
            try:
                change.execute()
                return "succeeded"
            except FileConflictError as error:
                self.assertIn("FILE_STALE_VERSION", str(error))
                return "conflict"

        with ThreadPoolExecutor(max_workers=2) as pool:
            outcomes = list(pool.map(commit, changes))

        self.assertEqual(sorted(outcomes), ["conflict", "succeeded"])
        self.assertIn(
            (self.workspace / "new.txt").read_text(encoding="utf-8"),
            {"first", "second"},
        )

    def test_parallel_writes_can_share_a_missing_parent_directory(self) -> None:
        changes = (
            self.tools.prepare_write_file(
                {"path": "pkg/first.txt", "content": "first"}
            ),
            self.tools.prepare_write_file(
                {"path": "pkg/second.txt", "content": "second"}
            ),
        )
        shared_parent = self.tools.resolver.workspace / "pkg"

        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(lambda change: change.execute(), changes))

        self.assertEqual(
            (shared_parent / "first.txt").read_text(encoding="utf-8"), "first"
        )
        self.assertEqual(
            (shared_parent / "second.txt").read_text(encoding="utf-8"), "second"
        )
        self.assertEqual(
            sum(shared_parent.resolve().as_posix() in result.created_directories for result in results),
            1,
        )

    def test_failed_parent_creator_cannot_break_peer_before_peer_temp_creation(
        self,
    ) -> None:
        creator = self.tools.prepare_write_file(
            {"path": "pkg/creator.txt", "content": "creator"}
        )
        peer = self.tools.prepare_write_file(
            {"path": "pkg/peer.txt", "content": "peer"}
        )
        shared_parent = self.tools.resolver.workspace / "pkg"
        parent_created = threading.Event()
        peer_adopted_parent = threading.Event()
        allow_peer_temp = threading.Event()
        real_create_directory = DirectoryMutationLease.create_directory
        real_mkstemp = tempfile.mkstemp

        def creator_mkdir(lease, path: Path, mode: int) -> bool:
            created = real_create_directory(lease, path, mode)
            if Path(path) == shared_parent:
                parent_created.set()
                self.assertTrue(peer_adopted_parent.wait(timeout=2))
            return created

        def interleaved_mkstemp(*args, **kwargs):
            prefix = kwargs.get("prefix", "")
            if prefix.startswith(".peer.txt."):
                peer_adopted_parent.set()
                self.assertTrue(allow_peer_temp.wait(timeout=2))
            elif prefix.startswith(".creator.txt."):
                raise OSError("creator disk failure")
            return real_mkstemp(*args, **kwargs)

        with (
            patch.object(
                DirectoryMutationLease,
                "create_directory",
                autospec=True,
                side_effect=creator_mkdir,
            ),
            patch(
                "mca.tools.filesystem.tempfile.mkstemp",
                side_effect=interleaved_mkstemp,
            ),
            ThreadPoolExecutor(max_workers=2) as pool,
        ):
            creator_future = pool.submit(creator.execute)
            self.assertTrue(parent_created.wait(timeout=2))
            peer_future = pool.submit(peer.execute)
            with self.assertRaisesRegex(OSError, "creator disk failure"):
                creator_future.result(timeout=2)
            self.assertTrue(shared_parent.is_dir())
            allow_peer_temp.set()
            peer_result = peer_future.result(timeout=2)

        self.assertEqual(
            (shared_parent / "peer.txt").read_text(encoding="utf-8"), "peer"
        )
        self.assertEqual(
            peer_result.created_directories, (str(shared_parent.resolve()),)
        )

    def test_successful_peer_owns_parent_when_creator_fails_after_peer_commit(
        self,
    ) -> None:
        creator = self.tools.prepare_write_file(
            {"path": "pkg/creator.txt", "content": "creator"}
        )
        peer = self.tools.prepare_write_file(
            {"path": "pkg/peer.txt", "content": "peer"}
        )
        shared_parent = self.tools.resolver.workspace / "pkg"
        parent_created = threading.Event()
        peer_committed = threading.Event()
        real_create_directory = DirectoryMutationLease.create_directory
        real_mkstemp = tempfile.mkstemp

        def creator_mkdir(lease, path: Path, mode: int) -> bool:
            created = real_create_directory(lease, path, mode)
            if Path(path) == shared_parent:
                parent_created.set()
                self.assertTrue(peer_committed.wait(timeout=2))
            return created

        def fail_creator_temp(*args, **kwargs):
            prefix = kwargs.get("prefix", "")
            if prefix.startswith(".creator.txt."):
                raise OSError("creator disk failure")
            return real_mkstemp(*args, **kwargs)

        with (
            patch.object(
                DirectoryMutationLease,
                "create_directory",
                autospec=True,
                side_effect=creator_mkdir,
            ),
            patch(
                "mca.tools.filesystem.tempfile.mkstemp",
                side_effect=fail_creator_temp,
            ),
            ThreadPoolExecutor(max_workers=2) as pool,
        ):
            creator_future = pool.submit(creator.execute)
            self.assertTrue(parent_created.wait(timeout=2))
            peer_result = peer.execute()
            peer_committed.set()
            with self.assertRaisesRegex(OSError, "creator disk failure"):
                creator_future.result(timeout=2)

        self.assertEqual(
            (shared_parent / "peer.txt").read_text(encoding="utf-8"), "peer"
        )
        self.assertEqual(
            peer_result.created_directories, (str(shared_parent.resolve()),)
        )

    def test_prepare_write_accepts_a_missing_parent_directory(self) -> None:
        change = self.tools.prepare_write_file(
            {"path": "pkg/sub/module.py", "content": "x = 1\n"}
        )

        self.assertFalse((self.workspace / "pkg").exists())
        self.assertFalse(change.existed_before)

    def test_execute_creates_missing_parent_directories_and_reports_them(self) -> None:
        change = self.tools.prepare_write_file(
            {"path": "pkg/sub/module.py", "content": "x = 1\n"}
        )

        result = change.execute()

        target = self.workspace / "pkg" / "sub" / "module.py"
        self.assertEqual(target.read_text(encoding="utf-8"), "x = 1\n")
        self.assertEqual(stat.S_IMODE((self.workspace / "pkg").stat().st_mode), 0o755)
        self.assertEqual(
            result.created_directories,
            (str((self.workspace / "pkg").resolve()), str(target.parent.resolve())),
        )

    def test_execute_into_existing_directory_reports_no_created_directories(self) -> None:
        (self.workspace / "pkg").mkdir()
        change = self.tools.prepare_write_file(
            {"path": "pkg/module.py", "content": "x = 1\n"}
        )

        result = change.execute()

        self.assertEqual(result.created_directories, ())

    def test_failed_write_rolls_back_directories_it_created(self) -> None:
        change = self.tools.prepare_write_file(
            {"path": "pkg/sub/module.py", "content": "x = 1\n"}
        )

        with patch(
            "mca.tools.filesystem.os.replace", side_effect=OSError("disk full")
        ):
            with self.assertRaises(OSError):
                change.execute()

        self.assertFalse((self.workspace / "pkg").exists())

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

    def test_keyboard_interrupt_immediately_after_replace_reports_committed_change(self) -> None:
        path = self.write_bytes("committed.txt", b"old")
        change = self.tools.prepare_write_file(
            {"path": "committed.txt", "content": "new"}
        )
        real_replace = os.replace

        def replace_then_interrupt(source: object, target: object) -> None:
            real_replace(source, target)
            raise KeyboardInterrupt

        with patch(
            "mca.tools.filesystem.os.replace", side_effect=replace_then_interrupt
        ):
            result = change.execute()

        self.assertEqual(path.read_bytes(), b"new")
        self.assertEqual(result.after_hash, hashlib.sha256(b"new").hexdigest())
        self.assertIs(result.interruption_warning, True)
        self.assertEqual(result.after_mode, 0o644)

    def test_keyboard_interrupt_during_parent_fsync_marks_committed_change(self) -> None:
        path = self.write_bytes("fsync-interrupted.txt", b"old")
        path.chmod(0o640)
        change = self.tools.prepare_write_file(
            {"path": "fsync-interrupted.txt", "content": "new"}
        )

        with patch(
            "mca.tools.filesystem._fsync_directory", side_effect=KeyboardInterrupt
        ):
            result = change.execute()

        self.assertEqual(path.read_bytes(), b"new")
        self.assertEqual(result.after_mode, 0o640)
        self.assertIs(result.interruption_warning, True)

    @unittest.skipUnless(hasattr(signal, "pthread_sigmask"), "requires pthread_sigmask")
    def test_keyboard_interrupt_during_sigint_unmask_marks_committed_change(self) -> None:
        path = self.write_bytes("unmask-interrupted.txt", b"old")
        change = self.tools.prepare_write_file(
            {"path": "unmask-interrupted.txt", "content": "new"}
        )

        with patch(
            "mca.tools.filesystem.signal.pthread_sigmask",
            side_effect=[set(), KeyboardInterrupt],
        ):
            result = change.execute()

        self.assertEqual(path.read_bytes(), b"new")
        self.assertIs(result.interruption_warning, True)

    def test_system_exit_after_replace_is_not_swallowed(self) -> None:
        path = self.write_bytes("system-exit.txt", b"old")
        change = self.tools.prepare_write_file(
            {"path": "system-exit.txt", "content": "new"}
        )
        real_replace = os.replace

        def replace_then_exit(source: object, target: object) -> None:
            real_replace(source, target)
            raise SystemExit(17)

        with patch(
            "mca.tools.filesystem.os.replace", side_effect=replace_then_exit
        ):
            with self.assertRaises(SystemExit) as raised:
                change.execute()

        self.assertEqual(raised.exception.code, 17)
        self.assertEqual(path.read_bytes(), b"new")

    def test_keyboard_interrupt_before_replace_propagates_without_committing(self) -> None:
        path = self.write_bytes("uncommitted.txt", b"old")
        change = self.tools.prepare_write_file(
            {"path": "uncommitted.txt", "content": "new"}
        )

        with patch("mca.tools.filesystem.os.replace", side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                change.execute()

        self.assertEqual(path.read_bytes(), b"old")

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
