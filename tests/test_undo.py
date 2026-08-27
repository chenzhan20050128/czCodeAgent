"""Hash-checked managed-file undo tests."""

from __future__ import annotations

import base64
import hashlib
import os
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

    def undo(self, state: SessionState | None = None, store: RolloutStore | None = None):
        return ManagedUndo(
            store or self.store, state or self.state, self.workspace
        ).undo_turn(self.turn_id)


class ManagedUndoTests(UndoTestCase):
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
        self.append(
            "file_snapshot",
            {
                "turn_id": self.turn_id,
                "path": str(path),
                "existed_before": True,
                "before_bytes": base64.b64encode(b"old").decode(),
                "before_encoding": "base64",
                "before_mode": 0o644,
            },
        )

        result = self.undo()

        self.assertEqual(result.status, "conflict")
        self.assertEqual(result.files[0].status, "ineligible")
        self.assertEqual(path.read_text(encoding="utf-8"), "live")

    def test_refuses_symlink_directory_and_outside_paths_before_mutation(self) -> None:
        safe = self.workspace / "safe.txt"
        safe.write_text("changed", encoding="utf-8")
        safe_snapshot = {
            "turn_id": self.turn_id,
            "path": str(safe),
            "existed_before": True,
            "before_bytes": base64.b64encode(b"old").decode(),
            "before_encoding": "base64",
            "before_mode": 0o644,
            "after_hash": hashlib.sha256(b"changed").hexdigest(),
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
        real_replace = os.replace
        calls = 0

        def fail_second_replace(source: object, target: object) -> None:
            nonlocal calls
            calls += 1
            if calls == 2:
                raise OSError("disk failure")
            real_replace(source, target)

        with patch("mca.undo.os.replace", side_effect=fail_second_replace):
            result = self.undo()

        self.assertEqual(result.status, "partial")
        self.assertEqual([item.status for item in result.files], ["restored", "failed"])
        self.assertEqual(first.read_text(encoding="utf-8"), "a")
        self.assertEqual(second.read_text(encoding="utf-8"), "changed-b")
        self.assertEqual(self.store.load()[-1].payload["status"], "partial")

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

    def test_invalid_base64_snapshot_fails_preflight_without_mutation(self) -> None:
        path = self.workspace / "bad.txt"
        path.write_text("current", encoding="utf-8")
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
            },
        )

        result = self.undo()

        self.assertEqual(result.status, "conflict")
        self.assertEqual(result.files[0].status, "ineligible")
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
