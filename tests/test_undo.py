"""Hash-checked managed-file undo tests."""

from __future__ import annotations

import base64
import hashlib
import os
import shutil
import stat
import sys
import tempfile
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mca.domain import SessionReducer, SessionState
from mca.executor import AcceptedToolCall, ToolExecutor
from mca.store import RolloutStore
from mca.tools import create_tool_registry
from mca.undo import ManagedUndo, UndoError
from mca.approval import ApprovalDecision


class AllowApprover:
    def decide(self, request: object) -> ApprovalDecision:
        del request
        return ApprovalDecision.ALLOW_ONCE


class UndoTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.workspace = self.root / "work"
        self.workspace.mkdir()
        self.sessions = self.root / "sessions"
        self.session_id = str(uuid.uuid4())
        self.turn_id = str(uuid.uuid4())
        self.store = RolloutStore.create(self.sessions, self.session_id)
        self.addCleanup(self.store.close)
        self.state = SessionState()
        self.append(
            "session_created",
            {
                "cwd": str(self.workspace),
                "model": "test-model",
                "context_window": 4096,
            },
        )
        self.append(
            "turn_started", {"turn_id": self.turn_id, "user_input": "change files"}
        )
        self.executor = ToolExecutor(
            create_tool_registry(self.workspace),
            self.store,
            self.state,
            AllowApprover(),
            self.workspace,
        )

    def append(self, event_type: str, payload: dict[str, object]) -> None:
        event = self.store.append(event_type, payload)
        SessionReducer.apply(self.state, event)

    def write_call(self, call_id: str, path: str, content: str) -> None:
        import json

        raw = json.dumps({"path": path, "content": content})
        self.append(
            "assistant_accepted",
            {
                "tool_calls": [
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {"name": "write_file", "arguments": raw},
                    }
                ]
            },
        )
        self.executor.execute(
            AcceptedToolCall(
                call_key=f"{self.state.last_seq}:{call_id}",
                provider_call_id=call_id,
                name="write_file",
                raw_arguments=raw,
            )
        )

    def pending_write_call(self, call_id: str = "snapshot") -> str:
        self.append(
            "assistant_accepted",
            {
                "tool_calls": [
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {"name": "write_file", "arguments": "{}"},
                    }
                ]
            },
        )
        return f"{self.state.last_seq}:{call_id}"

    def undo(self, state: SessionState | None = None, store: RolloutStore | None = None):
        return ManagedUndo(
            store or self.store, state or self.state, self.workspace
        ).undo_turn(self.turn_id)


class ManagedUndoTests(UndoTestCase):
    def test_quarantine_name_is_deterministic_and_operation_scoped(self) -> None:
        from mca.undo import _deterministic_quarantine_name

        path = self.workspace / "nested" / "file.txt"
        expected_hash = hashlib.sha256(b"managed").hexdigest()

        first = _deterministic_quarantine_name(
            path.name, self.turn_id, path, expected_hash
        )
        second = _deterministic_quarantine_name(
            path.name, self.turn_id, path, expected_hash
        )
        other = _deterministic_quarantine_name(
            path.name, str(uuid.uuid4()), path, expected_hash
        )

        self.assertEqual(first, second)
        self.assertRegex(first, r"^\.file\.txt\.mca-undo-[0-9a-f]{16}$")
        self.assertNotEqual(first, other)

    def test_existing_file_retry_recovers_after_crash_immediately_after_quarantine(self) -> None:
        class SimulatedCrash(BaseException):
            pass

        path = self.workspace / "existing-crash.txt"
        path.write_text("before", encoding="utf-8")
        path.chmod(0o640)
        self.write_call("write", "existing-crash.txt", "after")
        real_rename = os.rename

        def rename_then_crash(*args: object, **kwargs: object) -> None:
            real_rename(*args, **kwargs)
            raise SimulatedCrash

        with patch("mca.undo.os.rename", side_effect=rename_then_crash):
            with self.assertRaises(SimulatedCrash):
                self.undo()

        quarantines = list(self.workspace.glob(".existing-crash.txt.mca-undo-*"))
        self.assertEqual(len(quarantines), 1)
        self.assertFalse(path.exists())
        self.store.close()
        with RolloutStore.open(self.sessions, self.session_id) as reopened:
            replayed = SessionReducer.replay(reopened.load())
            result = ManagedUndo(reopened, replayed, self.workspace).undo_turn(
                self.turn_id
            )

        self.assertEqual(result.status, "succeeded")
        self.assertEqual(result.files[0].status, "restored")
        self.assertEqual(path.read_text(encoding="utf-8"), "before")
        self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o640)
        self.assertEqual(list(self.workspace.glob(".existing-crash.txt.mca-undo-*")), [])

    def test_new_file_retry_recovers_after_crash_immediately_after_quarantine(self) -> None:
        class SimulatedCrash(BaseException):
            pass

        path = self.workspace / "new-crash.txt"
        self.write_call("write", "new-crash.txt", "after")
        real_rename = os.rename

        def rename_then_crash(*args: object, **kwargs: object) -> None:
            real_rename(*args, **kwargs)
            raise SimulatedCrash

        with patch("mca.undo.os.rename", side_effect=rename_then_crash):
            with self.assertRaises(SimulatedCrash):
                self.undo()

        quarantines = list(self.workspace.glob(".new-crash.txt.mca-undo-*"))
        self.assertEqual(len(quarantines), 1)
        self.assertFalse(path.exists())
        self.store.close()
        with RolloutStore.open(self.sessions, self.session_id) as reopened:
            replayed = SessionReducer.replay(reopened.load())
            result = ManagedUndo(reopened, replayed, self.workspace).undo_turn(
                self.turn_id
            )

        self.assertEqual(result.status, "succeeded")
        self.assertEqual(result.files[0].status, "deleted")
        self.assertFalse(path.exists())
        self.assertEqual(list(self.workspace.glob(".new-crash.txt.mca-undo-*")), [])

    def test_completed_restore_with_quarantine_is_cleaned_as_already_restored(self) -> None:
        from mca.undo import _deterministic_quarantine_name

        path = self.workspace / "completed.txt"
        path.write_text("before", encoding="utf-8")
        path.chmod(0o640)
        self.write_call("write", "completed.txt", "after")
        snapshot = self.state.file_snapshots[(self.turn_id, str(path.resolve()))]
        quarantine = path.with_name(
            _deterministic_quarantine_name(
                path.name, self.turn_id, path.resolve(), snapshot.after_hash
            )
        )
        path.rename(quarantine)
        path.write_text("before", encoding="utf-8")
        path.chmod(0o640)

        result = self.undo()

        self.assertEqual(result.status, "succeeded")
        self.assertEqual(result.files[0].status, "already_restored")
        self.assertFalse(quarantine.exists())

    def test_unexpected_quarantine_hash_conflicts_without_touching_target(self) -> None:
        from mca.undo import _deterministic_quarantine_name

        path = self.workspace / "quarantine-conflict.txt"
        path.write_text("before", encoding="utf-8")
        self.write_call("write", "quarantine-conflict.txt", "after")
        snapshot = self.state.file_snapshots[(self.turn_id, str(path.resolve()))]
        quarantine = path.with_name(
            _deterministic_quarantine_name(
                path.name, self.turn_id, path.resolve(), snapshot.after_hash
            )
        )
        quarantine.write_text("unexpected", encoding="utf-8")

        result = self.undo()

        self.assertEqual(result.status, "conflict")
        self.assertEqual(path.read_text(encoding="utf-8"), "after")
        self.assertEqual(quarantine.read_text(encoding="utf-8"), "unexpected")

    def test_external_chmod_after_write_is_an_undo_conflict(self) -> None:
        path = self.workspace / "chmod.txt"
        path.write_text("before", encoding="utf-8")
        path.chmod(0o640)
        self.write_call("write", "chmod.txt", "after")
        path.chmod(0o600)

        result = self.undo()

        self.assertEqual(result.status, "conflict")
        self.assertEqual(result.files[0].status, "conflict")
        self.assertEqual(path.read_text(encoding="utf-8"), "after")
        self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)

    def test_restores_original_bytes_and_mode_atomically(self) -> None:
        path = self.workspace / "existing.txt"
        original = b"original\n"
        path.write_bytes(original)
        path.chmod(0o640)
        self.write_call("write", "existing.txt", "changed\n")

        result = self.undo()

        self.assertEqual(path.read_bytes(), original)
        self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o640)
        self.assertEqual(result.status, "succeeded")
        self.assertEqual(result.files[0].status, "restored")
        event = self.store.load()[-1]
        self.assertEqual(event.type, "undo_finished")
        self.assertEqual(event.payload["status"], "succeeded")

    def test_deletes_new_file_when_latest_hash_still_matches(self) -> None:
        path = self.workspace / "new.txt"
        self.write_call("write", "new.txt", "created")

        result = self.undo()

        self.assertFalse(path.exists())
        self.assertEqual(result.files[0].status, "deleted")

    def test_multiple_edits_restore_first_baseline(self) -> None:
        path = self.workspace / "many.txt"
        path.write_text("first", encoding="utf-8")
        self.write_call("one", "many.txt", "second")
        self.write_call("two", "many.txt", "third")

        result = self.undo()

        self.assertEqual(path.read_text(encoding="utf-8"), "first")
        self.assertEqual(len(result.files), 1)
        self.assertEqual(result.files[0].status, "restored")

    def test_any_hash_conflict_prevents_all_mutations(self) -> None:
        first = self.workspace / "first.txt"
        second = self.workspace / "second.txt"
        first.write_text("one", encoding="utf-8")
        second.write_text("two", encoding="utf-8")
        self.write_call("one", "first.txt", "changed-one")
        self.write_call("two", "second.txt", "changed-two")
        second.write_text("external", encoding="utf-8")

        result = self.undo()

        self.assertEqual(result.status, "conflict")
        self.assertEqual(first.read_text(encoding="utf-8"), "changed-one")
        self.assertEqual(second.read_text(encoding="utf-8"), "external")
        self.assertEqual(
            {item.status for item in result.files}, {"not_modified", "conflict"}
        )
        event = self.store.load()[-1]
        self.assertEqual(event.type, "undo_finished")
        self.assertEqual(event.payload["status"], "conflict")

    def test_missing_latest_hash_is_ineligible_and_does_not_mutate(self) -> None:
        path = self.workspace / "untouched.txt"
        path.write_text("live", encoding="utf-8")
        call_key = self.pending_write_call()
        self.append(
            "file_snapshot",
            {
                "turn_id": self.turn_id,
                "path": str(path.resolve()),
                "existed_before": True,
                "before_bytes": base64.b64encode(b"old").decode(),
                "before_encoding": "base64",
                "before_mode": 0o644,
                "call_key": call_key,
            },
        )

        result = self.undo()

        self.assertEqual(result.status, "conflict")
        self.assertEqual(result.files[0].status, "ineligible")
        self.assertEqual(path.read_text(encoding="utf-8"), "live")

    def test_refuses_symlink_directory_and_outside_paths_before_mutation(self) -> None:
        safe = self.workspace / "safe.txt"
        safe.write_text("changed", encoding="utf-8")
        call_key = self.pending_write_call()
        safe_snapshot = {
            "turn_id": self.turn_id,
            "path": str(safe),
            "existed_before": True,
            "before_bytes": base64.b64encode(b"old").decode(),
            "before_encoding": "base64",
            "before_mode": 0o644,
            "after_hash": hashlib.sha256(b"changed").hexdigest(),
            "after_mode": 0o644,
            "call_key": call_key,
        }
        self.append("file_snapshot", safe_snapshot)
        outside = self.root / "outside.txt"
        outside.write_text("outside", encoding="utf-8")
        self.append(
            "file_snapshot",
            {
                **safe_snapshot,
                "path": str(outside),
                "after_hash": hashlib.sha256(b"outside").hexdigest(),
            },
        )

        result = self.undo()

        self.assertEqual(result.status, "conflict")
        self.assertEqual(safe.read_text(encoding="utf-8"), "changed")
        self.assertEqual(outside.read_text(encoding="utf-8"), "outside")
        self.assertIn("outside workspace", result.files[1].detail)

    def test_replayed_lexical_parent_escape_cannot_modify_outside_file(self) -> None:
        outside = self.root / "outside.txt"
        outside.write_text("external", encoding="utf-8")
        disguised = self.workspace / ".." / "outside.txt"
        call_key = self.pending_write_call()
        self.append(
            "file_snapshot",
            {
                "turn_id": self.turn_id,
                "path": str(disguised),
                "existed_before": True,
                "before_bytes": base64.b64encode(b"baseline").decode(),
                "before_encoding": "base64",
                "before_mode": 0o644,
                "after_hash": hashlib.sha256(b"external").hexdigest(),
                "after_mode": 0o644,
                "call_key": call_key,
            },
        )

        result = self.undo()

        self.assertEqual(result.status, "conflict")
        self.assertEqual(result.files[0].status, "ineligible")
        self.assertIn("canonical", result.files[0].detail)
        self.assertEqual(outside.read_text(encoding="utf-8"), "external")

    def test_parent_symlink_swap_before_restore_cannot_write_outside(self) -> None:
        managed_dir = self.workspace / "dir"
        managed_dir.mkdir()
        managed = managed_dir / "file.txt"
        managed.write_text("before", encoding="utf-8")
        self.write_call("write", "dir/file.txt", "after")
        outside_dir = self.root / "outside"
        outside_dir.mkdir()
        outside = outside_dir / "file.txt"
        outside.write_text("outside", encoding="utf-8")
        real_open_parent = __import__("mca.undo", fromlist=["_open_parent_fd"])._open_parent_fd
        calls = 0

        def swap_then_open(workspace: Path, path: Path) -> int:
            nonlocal calls
            calls += 1
            if calls == 2:
                shutil.move(managed_dir, self.root / "original-dir")
                managed_dir.symlink_to(outside_dir, target_is_directory=True)
            return real_open_parent(workspace, path)

        with patch("mca.undo._open_parent_fd", side_effect=swap_then_open):
            result = self.undo()

        self.assertEqual(result.status, "conflict")
        self.assertEqual(result.files[0].status, "conflict")
        self.assertEqual(outside.read_text(encoding="utf-8"), "outside")

    def test_parent_symlink_swap_before_delete_cannot_delete_outside(self) -> None:
        managed_dir = self.workspace / "dir"
        managed_dir.mkdir()
        managed = managed_dir / "new.txt"
        self.write_call("write", "dir/new.txt", "created")
        outside_dir = self.root / "outside"
        outside_dir.mkdir()
        outside = outside_dir / "new.txt"
        outside.write_text("outside", encoding="utf-8")
        real_open_parent = __import__("mca.undo", fromlist=["_open_parent_fd"])._open_parent_fd
        calls = 0

        def swap_then_open(workspace: Path, path: Path) -> int:
            nonlocal calls
            calls += 1
            if calls == 2:
                shutil.move(managed_dir, self.root / "original-dir")
                managed_dir.symlink_to(outside_dir, target_is_directory=True)
            return real_open_parent(workspace, path)

        with patch("mca.undo._open_parent_fd", side_effect=swap_then_open):
            result = self.undo()

        self.assertEqual(result.status, "conflict")
        self.assertEqual(result.files[0].status, "conflict")
        self.assertEqual(outside.read_text(encoding="utf-8"), "outside")

    def test_concurrent_replacement_after_quarantine_is_never_overwritten(self) -> None:
        managed_dir = self.workspace / "dir"
        managed_dir.mkdir()
        path = managed_dir / "file.txt"
        path.write_text("baseline", encoding="utf-8")
        self.write_call("write", "dir/file.txt", "managed")
        real_rename = os.rename
        quarantine_names: list[str] = []

        def rename_then_replace(*args: object, **kwargs: object) -> None:
            real_rename(*args, **kwargs)
            if not quarantine_names:
                quarantine_names.append(str(args[1]))
                path.write_text("concurrent", encoding="utf-8")

        with patch("mca.undo.os.rename", side_effect=rename_then_replace):
            result = self.undo()

        self.assertEqual(result.status, "conflict")
        self.assertEqual(result.files[0].status, "conflict")
        self.assertIn(quarantine_names[0], result.files[0].detail)
        self.assertEqual(path.read_text(encoding="utf-8"), "concurrent")
        quarantine = managed_dir / quarantine_names[0]
        self.assertEqual(quarantine.read_text(encoding="utf-8"), "managed")

    def test_concurrent_replacement_during_new_file_undo_is_not_deleted(self) -> None:
        managed_dir = self.workspace / "dir"
        managed_dir.mkdir()
        path = managed_dir / "new.txt"
        self.write_call("write", "dir/new.txt", "managed")
        real_rename = os.rename

        def rename_then_replace(*args: object, **kwargs: object) -> None:
            real_rename(*args, **kwargs)
            path.write_text("concurrent", encoding="utf-8")

        with patch("mca.undo.os.rename", side_effect=rename_then_replace):
            result = self.undo()

        self.assertEqual(result.status, "conflict")
        self.assertEqual(result.files[0].status, "conflict")
        self.assertEqual(path.read_text(encoding="utf-8"), "concurrent")
        quarantine = list(managed_dir.glob(".new.txt.mca-undo-*"))
        self.assertEqual(len(quarantine), 1)
        self.assertEqual(quarantine[0].read_text(encoding="utf-8"), "managed")
        self.assertIn(quarantine[0].name, result.files[0].detail)

    def test_resume_reconstructed_state_can_undo(self) -> None:
        path = self.workspace / "resume.txt"
        path.write_text("before", encoding="utf-8")
        self.write_call("write", "resume.txt", "after")
        self.store.close()

        with RolloutStore.open(self.sessions, self.session_id) as reopened:
            replayed = SessionReducer.replay(reopened.load())
            result = ManagedUndo(reopened, replayed, self.workspace).undo_turn(
                self.turn_id
            )

        self.assertEqual(result.status, "succeeded")
        self.assertEqual(path.read_text(encoding="utf-8"), "before")

    def test_partial_io_failure_reports_each_file_and_persists_partial(self) -> None:
        first = self.workspace / "a.txt"
        second = self.workspace / "b.txt"
        first.write_text("a", encoding="utf-8")
        second.write_text("b", encoding="utf-8")
        self.write_call("one", "a.txt", "changed-a")
        self.write_call("two", "b.txt", "changed-b")
        real_link = os.link
        calls = 0

        def fail_second_install(*args: object, **kwargs: object) -> None:
            nonlocal calls
            calls += 1
            if calls == 2:
                raise OSError("disk failure")
            real_link(*args, **kwargs)

        with patch("mca.undo.os.link", side_effect=fail_second_install):
            result = self.undo()

        self.assertEqual(result.status, "partial")
        self.assertEqual([item.status for item in result.files], ["restored", "failed"])
        self.assertEqual(first.read_text(encoding="utf-8"), "a")
        self.assertEqual(second.read_text(encoding="utf-8"), "changed-b")
        self.assertEqual(self.store.load()[-1].payload["status"], "partial")

    def test_partial_undo_retries_only_remaining_file_then_becomes_idempotent(self) -> None:
        first = self.workspace / "a.txt"
        second = self.workspace / "b.txt"
        first.write_text("a", encoding="utf-8")
        second.write_text("b", encoding="utf-8")
        self.write_call("one", "a.txt", "changed-a")
        self.write_call("two", "b.txt", "changed-b")
        real_link = os.link
        calls = 0

        def fail_second_install(*args: object, **kwargs: object) -> None:
            nonlocal calls
            calls += 1
            if calls == 2:
                raise OSError("transient failure")
            real_link(*args, **kwargs)

        with patch("mca.undo.os.link", side_effect=fail_second_install):
            first_result = self.undo()

        second_result = self.undo()
        event_count = len(self.store.load())
        third_result = self.undo()

        self.assertEqual(first_result.status, "partial")
        self.assertEqual(second_result.status, "succeeded")
        self.assertEqual(
            [item.status for item in second_result.files],
            ["already_restored", "restored"],
        )
        self.assertEqual(first.read_text(encoding="utf-8"), "a")
        self.assertEqual(second.read_text(encoding="utf-8"), "b")
        self.assertEqual(third_result, second_result)
        self.assertEqual(len(self.store.load()), event_count)
        self.assertEqual(
            [event.payload["status"] for event in self.store.load() if event.type == "undo_finished"],
            ["partial", "succeeded"],
        )

    def test_new_file_deleted_before_fsync_failure_is_completed_on_retry(self) -> None:
        path = self.workspace / "new.txt"
        self.write_call("write", "new.txt", "created")

        with patch("mca.undo._fsync_parent", side_effect=OSError("fsync failed")):
            first = self.undo()
        second = self.undo()

        self.assertEqual(first.status, "partial")
        self.assertFalse(path.exists())
        self.assertEqual(second.status, "succeeded")
        self.assertEqual(second.files[0].status, "already_deleted")

    def test_oversized_current_file_is_ineligible_without_unbounded_read(self) -> None:
        path = self.workspace / "large.txt"
        path.write_text("before", encoding="utf-8")
        self.write_call("write", "large.txt", "after")
        path.write_bytes(b"x" * 65)

        result = ManagedUndo(
            self.store, self.state, self.workspace, max_file_bytes=64
        ).undo_turn(self.turn_id)

        self.assertEqual(result.status, "conflict")
        self.assertEqual(result.files[0].status, "ineligible")
        self.assertIn("size limit", result.files[0].detail)

    def test_oversized_snapshot_baseline_is_ineligible(self) -> None:
        path = self.workspace / "large-baseline.txt"
        path.write_text("current", encoding="utf-8")
        call_key = self.pending_write_call()
        self.append(
            "file_snapshot",
            {
                "turn_id": self.turn_id,
                "path": str(path.resolve()),
                "existed_before": True,
                "before_bytes": base64.b64encode(b"x" * 65).decode(),
                "before_encoding": "base64",
                "before_mode": 0o644,
                "after_hash": hashlib.sha256(b"current").hexdigest(),
                "after_mode": 0o644,
                "call_key": call_key,
            },
        )

        result = ManagedUndo(
            self.store, self.state, self.workspace, max_file_bytes=64
        ).undo_turn(self.turn_id)

        self.assertEqual(result.status, "conflict")
        self.assertEqual(result.files[0].status, "ineligible")
        self.assertIn("baseline exceeds size limit", result.files[0].detail)
        self.assertEqual(path.read_text(encoding="utf-8"), "current")

    def test_baseline_bytes_with_wrong_mode_are_not_treated_as_already_restored(self) -> None:
        path = self.workspace / "mode.txt"
        path.write_text("before", encoding="utf-8")
        path.chmod(0o640)
        self.write_call("write", "mode.txt", "after")
        path.write_text("before", encoding="utf-8")
        path.chmod(0o600)

        result = self.undo()

        self.assertEqual(result.status, "conflict")
        self.assertEqual(result.files[0].status, "conflict")
        self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)

    def test_double_undo_is_idempotent_without_duplicate_event(self) -> None:
        path = self.workspace / "once.txt"
        path.write_text("before", encoding="utf-8")
        self.write_call("write", "once.txt", "after")
        first = self.undo()
        event_count = len(self.store.load())

        second = self.undo()

        self.assertEqual(second, first)
        self.assertEqual(len(self.store.load()), event_count)
        self.assertEqual(path.read_text(encoding="utf-8"), "before")

    def test_invalid_base64_snapshot_is_rejected_before_undo(self) -> None:
        path = self.workspace / "bad.txt"
        path.write_text("current", encoding="utf-8")
        call_key = self.pending_write_call()
        before_events = len(self.store.load())

        with self.assertRaisesRegex(ValueError, "valid base64"):
            self.append(
                "file_snapshot",
                {
                    "turn_id": self.turn_id,
                    "path": str(path),
                    "existed_before": True,
                    "before_bytes": "not-base64!!",
                    "before_encoding": "base64",
                    "before_mode": 0o644,
                    "after_hash": hashlib.sha256(b"current").hexdigest(),
                    "after_mode": 0o644,
                    "call_key": call_key,
                },
            )

        self.assertEqual(len(self.store.load()), before_events + 1)
        self.assertEqual(path.read_text(encoding="utf-8"), "current")

    def test_unknown_turn_is_rejected_without_event(self) -> None:
        before = len(self.store.load())
        with self.assertRaisesRegex(UndoError, "unknown turn"):
            ManagedUndo(self.store, self.state, self.workspace).undo_turn(
                str(uuid.uuid4())
            )
        self.assertEqual(len(self.store.load()), before)


if __name__ == "__main__":
    unittest.main()
