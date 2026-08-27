"""Session resume and uncertain-side-effect reconciliation tests."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mca.domain import SessionReducer, SessionState, ToolStatus, TurnStatus
from mca.projection import (
    ProjectionBlockedError,
    ProjectionEnvironment,
    PromptProjector,
    validate_conversation,
)
from mca.session import (
    ReconciliationError,
    ResumeError,
    continuable_turn_id,
    reconcile_tool,
    resume_session,
)
from mca.store import (
    RolloutCorruptionError,
    RolloutStore,
    SessionLockedError,
)


def tool_call(call_id: str, name: str = "bash") -> dict[str, object]:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": "{}"},
    }


class ResumeTestCase(unittest.TestCase):
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
        self.state = SessionState()
        self.append(
            "session_created",
            {
                "cwd": str(self.workspace),
                "model": "test-model",
                "context_window": 4096,
            },
        )

    def tearDown(self) -> None:
        self.store.close()
        self.temporary.cleanup()

    def append(self, event_type: str, payload: dict[str, object]) -> None:
        event = self.store.append(event_type, payload)
        SessionReducer.apply(self.state, event)

    def start_turn(self, calls: tuple[str, ...] = ()) -> None:
        self.append(
            "turn_started",
            {"turn_id": self.turn_id, "user_input": "recover this task"},
        )
        if calls:
            self.append(
                "assistant_accepted",
                {"content": None, "tool_calls": [tool_call(call) for call in calls]},
            )

    def reopen(self):
        self.store.close()
        return resume_session(self.sessions, self.session_id, self.workspace)

    def environment(self) -> ProjectionEnvironment:
        return ProjectionEnvironment(
            cwd=str(self.workspace),
            platform="test",
            date="2026-08-27",
            is_git=False,
        )


class SessionResumeTests(ResumeTestCase):
    def test_completed_session_replays_without_appending_recovery_events(self) -> None:
        self.start_turn()
        self.append(
            "assistant_accepted",
            {"content": "done", "tool_calls": []},
        )
        self.append(
            "turn_finished",
            {"turn_id": self.turn_id, "status": "completed"},
        )
        before = self.store.load()

        with self.reopen() as resumed:
            self.assertEqual(resumed.store.load(), before)
            self.assertEqual(resumed.state, SessionReducer.replay(before))
            self.assertIsNone(continuable_turn_id(resumed.state))

    def test_requested_call_is_durably_closed_as_not_executed(self) -> None:
        self.start_turn(("pending",))

        with self.reopen() as resumed:
            call = resumed.state.tool_calls["3:pending"]
            self.assertIs(call.status, ToolStatus.NOT_EXECUTED)
            self.assertFalse(resumed.state.recovery_blocked)
            self.assertIsNone(continuable_turn_id(resumed.state))
            self.assertIsNone(resumed.state.active_turn_id)
            self.assertIs(
                resumed.state.turns[self.turn_id], TurnStatus.INTERRUPTED
            )
            self.assertEqual(resumed.store.load()[-2].type, "tool_finished")
            self.assertEqual(
                resumed.store.load()[-2].payload["status"], "not_executed"
            )
            self.assertEqual(resumed.store.load()[-1].type, "turn_finished")
            self.assertEqual(resumed.store.load()[-1].payload["status"], "interrupted")
            self.assertEqual(
                resumed.state, SessionReducer.replay(resumed.store.load())
            )
            messages = PromptProjector.project(
                resumed.store.load(), resumed.state, self.environment()
            )
            validate_conversation(messages)

    def test_started_call_becomes_unknown_and_blocks_projection(self) -> None:
        self.start_turn(("started",))
        self.append(
            "tool_started",
            {"call_key": "3:started", "call_id": "started"},
        )

        with self.reopen() as resumed:
            call = resumed.state.tool_calls["3:started"]
            self.assertIs(call.status, ToolStatus.OUTCOME_UNKNOWN)
            self.assertTrue(call.recovery_blocked)
            self.assertTrue(resumed.state.recovery_blocked)
            self.assertIs(
                resumed.state.turns[self.turn_id], TurnStatus.RECOVERY_BLOCKED
            )
            with self.assertRaisesRegex(ReconciliationError, "blocked"):
                continuable_turn_id(resumed.state)
            with self.assertRaises(ProjectionBlockedError):
                PromptProjector.project(
                    resumed.store.load(), resumed.state, self.environment()
                )

    def test_resume_rejects_noncanonical_uuid_and_missing_or_mismatched_cwd(
        self,
    ) -> None:
        self.store.close()
        with self.assertRaisesRegex(ValueError, "canonical UUID"):
            resume_session(self.sessions, self.session_id.upper(), self.workspace)

        missing = self.root / "missing"
        with self.assertRaisesRegex(ResumeError, "workspace.*exist"):
            resume_session(self.sessions, self.session_id, missing)

        other = self.root / "other"
        other.mkdir()
        with self.assertRaisesRegex(ResumeError, "cwd.*match"):
            resume_session(self.sessions, self.session_id, other)

        with resume_session(
            self.sessions, self.session_id, self.workspace
        ) as resumed:
            self.assertEqual(resumed.state.cwd, str(self.workspace))

    def test_lock_is_held_for_the_resumed_session_lifetime(self) -> None:
        self.store.close()
        first = resume_session(self.sessions, self.session_id, self.workspace)
        try:
            with self.assertRaises(SessionLockedError):
                resume_session(self.sessions, self.session_id, self.workspace)
        finally:
            first.close()

        with resume_session(self.sessions, self.session_id, self.workspace):
            pass

    def test_corruption_is_not_hidden_by_resume_wrapper(self) -> None:
        self.store.close()
        path = self.sessions / f"{self.session_id}.jsonl"
        lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
        path.write_text("{not json}\n" + "".join(lines), encoding="utf-8")

        with self.assertRaises(RolloutCorruptionError):
            resume_session(self.sessions, self.session_id, self.workspace)


class ReconciliationTests(ResumeTestCase):
    def resume_unknowns(self, *call_ids: str):
        self.start_turn(tuple(call_ids))
        for call_id in call_ids:
            self.append(
                "tool_started",
                {"call_key": f"3:{call_id}", "call_id": call_id},
            )
        return self.reopen()

    def test_success_and_failure_reconciliation_unblock_only_after_last_unknown(
        self,
    ) -> None:
        with self.resume_unknowns("a", "b") as resumed:
            first = reconcile_tool(
                resumed.store,
                resumed.state,
                "3:a",
                "succeeded",
                "verified file",
            )
            self.assertEqual(first.payload["outcome"], "succeeded")
            self.assertTrue(resumed.state.recovery_blocked)
            with self.assertRaisesRegex(ReconciliationError, "blocked"):
                continuable_turn_id(resumed.state)

            second = reconcile_tool(
                resumed.store,
                resumed.state,
                "3:b",
                "failed",
                "command failed",
            )
            self.assertEqual(second.payload["outcome"], "failed")
            self.assertFalse(resumed.state.recovery_blocked)
            self.assertEqual(continuable_turn_id(resumed.state), self.turn_id)
            self.assertIs(
                resumed.state.turns[self.turn_id], TurnStatus.ACTIVE
            )
            messages = PromptProjector.project(
                resumed.store.load(), resumed.state, self.environment()
            )
            self.assertEqual(
                [message["tool_call_id"] for message in messages if message["role"] == "tool"],
                ["a", "b"],
            )

    def test_abandon_reconciles_every_unknown_and_finishes_the_turn(self) -> None:
        with self.resume_unknowns("a", "b") as resumed:
            event = reconcile_tool(
                resumed.store,
                resumed.state,
                "3:a",
                "abandoned",
                "cannot verify safely",
            )

            self.assertEqual(event.payload["outcome"], "abandoned")
            self.assertFalse(resumed.state.recovery_blocked)
            self.assertIsNone(resumed.state.active_turn_id)
            self.assertIs(
                resumed.state.turns[self.turn_id], TurnStatus.ABANDONED
            )
            reconciled = [
                item for item in resumed.store.load() if item.type == "tool_reconciled"
            ]
            self.assertEqual(len(reconciled), 2)
            self.assertTrue(
                all(item.payload["outcome"] == "abandoned" for item in reconciled)
            )
            self.assertEqual(resumed.store.load()[-1].type, "turn_finished")
            self.assertEqual(resumed.store.load()[-1].payload["status"], "abandoned")
            self.assertIsNone(continuable_turn_id(resumed.state))

    def test_reconcile_rejects_invalid_inputs_and_non_unknown_call(self) -> None:
        self.start_turn(("pending",))
        with self.reopen() as resumed:
            before = resumed.store.load()
            invalid = (
                ("succeeded", 1),
                ("maybe", "note"),
            )
            for outcome, note in invalid:
                with self.subTest(outcome=outcome, note=note):
                    with self.assertRaises((TypeError, ReconciliationError)):
                        reconcile_tool(
                            resumed.store,
                            resumed.state,
                            "3:pending",
                            outcome,
                            note,  # type: ignore[arg-type]
                        )
            self.assertEqual(resumed.store.load(), before)

    def test_reconciliation_append_failure_does_not_unblock_memory(self) -> None:
        with self.resume_unknowns("a") as resumed:
            before = SessionReducer.replay(resumed.store.load())
            with patch.object(
                resumed.store, "append", side_effect=OSError("fsync failed")
            ):
                with self.assertRaisesRegex(ReconciliationError, "append"):
                    reconcile_tool(
                        resumed.store,
                        resumed.state,
                        "3:a",
                        "succeeded",
                        "checked",
                    )

            self.assertEqual(resumed.state, before)
            self.assertTrue(resumed.state.recovery_blocked)


if __name__ == "__main__":
    unittest.main()
