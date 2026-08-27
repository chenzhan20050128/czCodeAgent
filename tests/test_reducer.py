"""State-reducer and crash-recovery contract tests."""

from __future__ import annotations

import json
import sys
import unittest
import uuid
from dataclasses import FrozenInstanceError
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mca.domain import (
    DomainError,
    Event,
    FileSnapshot,
    SamplingOutcome,
    SessionReducer,
    SessionState,
    ToolCall,
    ToolStatus,
    TurnStatus,
    plan_recovery_events,
    reduce_event,
)


class ReducerTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.session_id = str(uuid.uuid4())
        self.turn_id = str(uuid.uuid4())
        self.state = SessionState()
        self.next_seq = 1

    def event(self, event_type: str, payload: dict[str, object]) -> Event:
        event = Event.create(
            seq=self.next_seq,
            session_id=self.session_id,
            event_type=event_type,
            payload=payload,
        )
        self.next_seq += 1
        return event

    def apply(self, event_type: str, payload: dict[str, object]) -> Event:
        event = self.event(event_type, payload)
        SessionReducer.apply(self.state, event)
        return event

    def create_session(self) -> None:
        self.apply(
            "session_created",
            {
                "cwd": "/workspace/project",
                "model": "deepseek-v4-flash",
                "context_window": 65_536,
            },
        )

    def start_turn(self) -> None:
        self.create_session()
        self.apply(
            "turn_started",
            {"turn_id": self.turn_id, "user_input": "fix the tests"},
        )

    @staticmethod
    def tool(call_id: str, name: str = "read_file") -> dict[str, object]:
        return {
            "id": call_id,
            "type": "function",
            "function": {"name": name, "arguments": json.dumps({})},
        }


class DomainValueTests(ReducerTestCase):
    def test_required_domain_values_are_enums_or_immutable_dataclasses(self) -> None:
        call = ToolCall(
            call_id="call-1",
            turn_id=self.turn_id,
            name="read_file",
            arguments="{}",
        )
        snapshot = FileSnapshot(
            turn_id=self.turn_id,
            path="src/app.py",
            existed_before=True,
            before_bytes="b2xk",
            before_mode=0o644,
        )

        with self.assertRaises(FrozenInstanceError):
            call.status = ToolStatus.STARTED  # type: ignore[misc]
        with self.assertRaises(FrozenInstanceError):
            snapshot.after_hash = "new"  # type: ignore[misc]
        self.assertEqual(SamplingOutcome.COMPLETE_TEXT.value, "complete_text")
        self.assertEqual(ToolStatus.OUTCOME_UNKNOWN.value, "outcome_unknown")
        self.assertEqual(TurnStatus.RECOVERY_BLOCKED.value, "recovery_blocked")

    def test_session_state_is_intentionally_mutable(self) -> None:
        self.state.cwd = "/new-workspace"
        self.assertEqual(self.state.cwd, "/new-workspace")


class SessionReducerTests(ReducerTestCase):
    def test_reduce_event_returns_independent_state_without_mutating_input(self) -> None:
        self.start_turn()
        before = SessionReducer.replay(self.state.events)
        assistant = self.event(
            "assistant_accepted",
            {"content": "done", "tool_calls": []},
        )

        derived = reduce_event(self.state, assistant)

        self.assertEqual(self.state, before)
        self.assertIsNot(derived, self.state)
        self.assertEqual(derived.last_seq, 3)
        self.assertEqual(derived.assistant_events, [assistant])
        for attribute in (
            "turns",
            "turn_inputs",
            "tool_calls",
            "file_snapshots",
            "assistant_events",
            "events",
        ):
            self.assertIsNot(
                getattr(derived, attribute), getattr(self.state, attribute)
            )
        derived.turns[self.turn_id] = TurnStatus.FAILED
        self.assertEqual(self.state.turns[self.turn_id], TurnStatus.ACTIVE)

    def test_reduce_event_leaves_input_unchanged_when_transition_fails(self) -> None:
        self.start_turn()
        before = SessionReducer.replay(self.state.events)
        invalid = Event.create(
            seq=self.state.last_seq + 1,
            session_id=self.session_id,
            event_type="unknown_event",
            payload={},
        )

        with self.assertRaises(DomainError):
            reduce_event(self.state, invalid)

        self.assertEqual(self.state, before)

    def test_session_created_establishes_identity_and_runtime_configuration(self) -> None:
        self.create_session()

        self.assertEqual(self.state.session_id, self.session_id)
        self.assertEqual(self.state.cwd, "/workspace/project")
        self.assertEqual(self.state.model, "deepseek-v4-flash")
        self.assertEqual(self.state.context_window, 65_536)
        self.assertEqual(self.state.last_seq, 1)

    def test_turn_started_sets_the_active_turn(self) -> None:
        self.start_turn()

        self.assertEqual(self.state.active_turn_id, self.turn_id)
        self.assertEqual(self.state.turns[self.turn_id], TurnStatus.ACTIVE)
        self.assertEqual(self.state.turn_inputs[self.turn_id], "fix the tests")

    def test_assistant_accepted_registers_all_requested_calls_in_order(self) -> None:
        self.start_turn()
        self.apply(
            "assistant_accepted",
            {
                "content": "I will inspect both files.",
                "finish_reason": "tool_calls",
                "tool_calls": [
                    self.tool("call-1"),
                    self.tool("call-2", "grep"),
                ],
            },
        )

        self.assertEqual(list(self.state.tool_calls), ["call-1", "call-2"])
        self.assertEqual(self.state.tool_calls["call-1"].status, ToolStatus.REQUESTED)
        self.assertEqual(self.state.tool_calls["call-2"].name, "grep")
        self.assertEqual(self.state.tool_calls["call-2"].turn_id, self.turn_id)

    def test_assistant_accepted_rejects_while_a_prior_call_is_unresolved(self) -> None:
        for prior_status in (ToolStatus.REQUESTED, ToolStatus.STARTED):
            with self.subTest(prior_status=prior_status):
                self.setUp()
                self.start_turn()
                self.apply(
                    "assistant_accepted",
                    {"tool_calls": [self.tool("call-1", "bash")]},
                )
                if prior_status is ToolStatus.STARTED:
                    self.apply("tool_started", {"call_id": "call-1"})
                last_seq = self.state.last_seq
                accepted_count = len(self.state.assistant_events)
                next_assistant = Event.create(
                    seq=last_seq + 1,
                    session_id=self.session_id,
                    event_type="assistant_accepted",
                    payload={"content": "continuing", "tool_calls": []},
                )

                with self.assertRaises(DomainError):
                    SessionReducer.apply(self.state, next_assistant)

                self.assertEqual(self.state.last_seq, last_seq)
                self.assertEqual(len(self.state.assistant_events), accepted_count)

    def test_assistant_accepted_rejects_while_recovery_is_blocked(self) -> None:
        self.start_turn()
        self.apply(
            "assistant_accepted",
            {"tool_calls": [self.tool("call-1", "bash")]},
        )
        self.apply("tool_started", {"call_id": "call-1"})
        self.apply(
            "tool_finished",
            {
                "call_id": "call-1",
                "status": "outcome_unknown",
                "result": "unknown",
                "recovery_blocked": True,
            },
        )
        last_seq = self.state.last_seq
        accepted_count = len(self.state.assistant_events)
        next_assistant = Event.create(
            seq=last_seq + 1,
            session_id=self.session_id,
            event_type="assistant_accepted",
            payload={"content": "continuing", "tool_calls": []},
        )

        with self.assertRaises(DomainError):
            SessionReducer.apply(self.state, next_assistant)

        self.assertEqual(self.state.last_seq, last_seq)
        self.assertEqual(len(self.state.assistant_events), accepted_count)
        self.assertTrue(self.state.recovery_blocked)

    def test_approval_start_and_finish_update_a_call_to_terminal(self) -> None:
        self.start_turn()
        self.apply(
            "assistant_accepted",
            {"tool_calls": [self.tool("call-1", "write_file")]},
        )
        self.apply(
            "approval_decided",
            {"call_id": "call-1", "scope": "once", "approved": True},
        )
        approved = self.state.tool_calls["call-1"]
        self.assertIs(approved.approved, True)
        self.assertEqual(approved.approval_scope, "once")

        self.apply("tool_started", {"call_id": "call-1"})
        self.assertEqual(
            self.state.tool_calls["call-1"].status, ToolStatus.STARTED
        )

        self.apply(
            "tool_finished",
            {
                "call_id": "call-1",
                "status": "succeeded",
                "result": "wrote file",
                "exit_code": 0,
                "truncated": False,
            },
        )
        finished = self.state.tool_calls["call-1"]
        self.assertEqual(finished.status, ToolStatus.SUCCEEDED)
        self.assertEqual(finished.result, "wrote file")
        self.assertEqual(finished.exit_code, 0)
        self.assertIs(finished.truncated, False)
        self.assertTrue(finished.is_terminal)

    def test_turn_finished_closes_the_active_turn(self) -> None:
        self.start_turn()
        self.apply(
            "turn_finished",
            {"turn_id": self.turn_id, "status": "completed"},
        )

        self.assertIsNone(self.state.active_turn_id)
        self.assertEqual(self.state.turns[self.turn_id], TurnStatus.COMPLETED)

    def test_latest_compaction_checkpoint_replaces_the_previous_one(self) -> None:
        self.start_turn()
        self.apply(
            "assistant_accepted",
            {"content": "checkpoint boundary", "tool_calls": []},
        )
        first = self.apply(
            "compaction_checkpoint",
            {
                "through_seq": 1,
                "summary": "first",
                "replacement_conversation": [{"role": "user", "content": "x"}],
            },
        )
        second = self.apply(
            "compaction_checkpoint",
            {
                "through_seq": 2,
                "summary": "second",
                "replacement_conversation": [],
            },
        )

        self.assertNotEqual(first, second)
        self.assertEqual(self.state.latest_checkpoint, second)

    def test_checkpoint_requires_an_active_turn(self) -> None:
        self.create_session()
        checkpoint = self.event(
            "compaction_checkpoint",
            {
                "through_seq": 1,
                "summary": "summary",
                "replacement_conversation": [],
            },
        )

        with self.assertRaises(DomainError):
            SessionReducer.apply(self.state, checkpoint)

        self.assertEqual(self.state.last_seq, 1)
        self.assertIsNone(self.state.latest_checkpoint)

    def test_checkpoint_requires_an_accepted_assistant_in_the_active_turn(self) -> None:
        self.start_turn()
        checkpoint = self.event(
            "compaction_checkpoint",
            {
                "through_seq": 2,
                "summary": "summary",
                "replacement_conversation": [],
            },
        )

        with self.assertRaises(DomainError):
            SessionReducer.apply(self.state, checkpoint)

        self.assertEqual(self.state.last_seq, 2)
        self.assertIsNone(self.state.latest_checkpoint)

    def test_checkpoint_does_not_reuse_an_assistant_from_an_earlier_turn(self) -> None:
        self.start_turn()
        self.apply(
            "assistant_accepted",
            {"content": "first turn complete", "tool_calls": []},
        )
        self.apply(
            "turn_finished",
            {"turn_id": self.turn_id, "status": "completed"},
        )
        second_turn_id = str(uuid.uuid4())
        self.apply(
            "turn_started",
            {"turn_id": second_turn_id, "user_input": "second task"},
        )
        last_seq = self.state.last_seq
        checkpoint = Event.create(
            seq=last_seq + 1,
            session_id=self.session_id,
            event_type="compaction_checkpoint",
            payload={
                "through_seq": last_seq,
                "summary": "summary",
                "replacement_conversation": [],
            },
        )

        with self.assertRaises(DomainError):
            SessionReducer.apply(self.state, checkpoint)

        self.assertEqual(self.state.last_seq, last_seq)
        self.assertIsNone(self.state.latest_checkpoint)

    def test_checkpoint_rejects_every_unresolved_tool_state(self) -> None:
        for unresolved_status in (
            ToolStatus.REQUESTED,
            ToolStatus.STARTED,
            ToolStatus.OUTCOME_UNKNOWN,
        ):
            with self.subTest(unresolved_status=unresolved_status):
                self.setUp()
                self.start_turn()
                self.apply(
                    "assistant_accepted",
                    {"tool_calls": [self.tool("call-1", "bash")]},
                )
                if unresolved_status is not ToolStatus.REQUESTED:
                    self.apply("tool_started", {"call_id": "call-1"})
                if unresolved_status is ToolStatus.OUTCOME_UNKNOWN:
                    self.apply(
                        "tool_finished",
                        {
                            "call_id": "call-1",
                            "status": "outcome_unknown",
                            "result": "unknown",
                            "recovery_blocked": True,
                        },
                    )
                last_seq = self.state.last_seq
                checkpoint = Event.create(
                    seq=last_seq + 1,
                    session_id=self.session_id,
                    event_type="compaction_checkpoint",
                    payload={
                        "through_seq": last_seq,
                        "summary": "summary",
                        "replacement_conversation": [],
                    },
                )

                with self.assertRaises(DomainError):
                    SessionReducer.apply(self.state, checkpoint)

                self.assertEqual(self.state.last_seq, last_seq)
                self.assertIsNone(self.state.latest_checkpoint)

    def test_file_snapshot_keeps_first_baseline_and_success_updates_after_hash(self) -> None:
        self.start_turn()
        self.apply(
            "assistant_accepted",
            {"tool_calls": [self.tool("write-1", "write_file")]},
        )
        first_snapshot = {
            "turn_id": self.turn_id,
            "path": "src/app.py",
            "existed_before": True,
            "before_bytes": "Zmlyc3Q=",
            "before_mode": 0o644,
        }
        self.apply("file_snapshot", first_snapshot)
        self.apply(
            "file_snapshot",
            {**first_snapshot, "before_bytes": "c2Vjb25k", "before_mode": 0o600},
        )
        self.apply("tool_started", {"call_id": "write-1"})
        self.apply(
            "tool_finished",
            {
                "call_id": "write-1",
                "status": "succeeded",
                "result": "ok",
                "path": "src/app.py",
                "after_hash": "sha256:new",
            },
        )

        snapshot = self.state.file_snapshots[(self.turn_id, "src/app.py")]
        self.assertEqual(snapshot.before_bytes, "Zmlyc3Q=")
        self.assertEqual(snapshot.before_mode, 0o644)
        self.assertEqual(snapshot.after_hash, "sha256:new")

    def test_tool_reconciled_closes_an_unknown_outcome(self) -> None:
        self.start_turn()
        self.apply(
            "assistant_accepted",
            {"tool_calls": [self.tool("call-1", "bash")]},
        )
        self.apply("tool_started", {"call_id": "call-1"})
        self.apply(
            "tool_finished",
            {
                "call_id": "call-1",
                "status": "outcome_unknown",
                "result": "process died after execution began",
                "recovery_blocked": True,
            },
        )
        self.assertTrue(self.state.recovery_blocked)

        self.apply(
            "tool_reconciled",
            {
                "call_id": "call-1",
                "outcome": "succeeded",
                "note": "verified the generated file",
            },
        )

        call = self.state.tool_calls["call-1"]
        self.assertEqual(call.status, ToolStatus.USER_CONFIRMED_SUCCESS)
        self.assertEqual(call.reconciliation_note, "verified the generated file")
        self.assertFalse(self.state.recovery_blocked)

    def test_unknown_outcome_blocks_turn_completion_until_reconciled(self) -> None:
        self.start_turn()
        self.apply(
            "assistant_accepted",
            {"tool_calls": [self.tool("call-1", "bash")]},
        )
        self.apply("tool_started", {"call_id": "call-1"})
        self.apply(
            "tool_finished",
            {
                "call_id": "call-1",
                "status": "outcome_unknown",
                "result": "execution outcome is unknown",
                "recovery_blocked": True,
            },
        )
        last_seq = self.state.last_seq
        event_count = len(self.state.events)
        blocked_turn = self.state.turns[self.turn_id]
        finish = Event.create(
            seq=last_seq + 1,
            session_id=self.session_id,
            event_type="turn_finished",
            payload={"turn_id": self.turn_id, "status": "completed"},
        )

        with self.assertRaises(DomainError):
            SessionReducer.apply(self.state, finish)

        self.assertEqual(self.state.last_seq, last_seq)
        self.assertEqual(len(self.state.events), event_count)
        self.assertEqual(self.state.active_turn_id, self.turn_id)
        self.assertEqual(self.state.turns[self.turn_id], blocked_turn)
        self.assertTrue(self.state.recovery_blocked)

    def test_recovery_planner_returns_deterministic_explicit_events(self) -> None:
        self.start_turn()
        self.apply(
            "assistant_accepted",
            {
                "tool_calls": [
                    self.tool("started-call", "bash"),
                    self.tool("pending-call", "read_file"),
                ]
            },
        )
        self.apply("tool_started", {"call_id": "started-call"})

        first_plan = plan_recovery_events(self.state)
        second_plan = plan_recovery_events(self.state)

        self.assertEqual(first_plan, second_plan)
        self.assertEqual([event.type for event in first_plan], ["tool_finished"] * 2)
        self.assertEqual([event.seq for event in first_plan], [5, 6])
        by_call = {event.payload["call_id"]: event for event in first_plan}
        self.assertEqual(
            by_call["started-call"].payload["status"], "outcome_unknown"
        )
        self.assertIs(by_call["started-call"].payload["recovery_blocked"], True)
        self.assertEqual(
            by_call["pending-call"].payload["status"], "not_executed"
        )
        self.assertEqual(
            self.state.tool_calls["started-call"].status, ToolStatus.STARTED
        )
        self.assertEqual(
            self.state.tool_calls["pending-call"].status, ToolStatus.REQUESTED
        )

        for event in first_plan:
            SessionReducer.apply(self.state, event)
        self.assertEqual(
            self.state.tool_calls["started-call"].status,
            ToolStatus.OUTCOME_UNKNOWN,
        )
        self.assertEqual(
            self.state.tool_calls["pending-call"].status,
            ToolStatus.NOT_EXECUTED,
        )
        self.assertTrue(self.state.recovery_blocked)

    def test_replay_reconstructs_the_same_state_from_facts(self) -> None:
        events = [
            self.event(
                "session_created",
                {"cwd": "/w", "model": "m", "context_window": 100},
            ),
            self.event(
                "turn_started",
                {"turn_id": self.turn_id, "user_input": "task"},
            ),
            self.event(
                "turn_finished",
                {"turn_id": self.turn_id, "status": "completed"},
            ),
        ]

        replayed = SessionReducer.replay(events)

        self.assertEqual(replayed.session_id, self.session_id)
        self.assertEqual(replayed.last_seq, 3)
        self.assertEqual(replayed.turns[self.turn_id], TurnStatus.COMPLETED)

    def test_invalid_transitions_raise_domain_error_without_advancing_sequence(self) -> None:
        invalid_before_creation = self.event(
            "turn_started", {"turn_id": self.turn_id}
        )
        with self.assertRaises(DomainError):
            SessionReducer.apply(self.state, invalid_before_creation)
        self.assertEqual(self.state.last_seq, 0)

        self.next_seq = 1
        self.start_turn()
        invalid_cases = [
            ("session_created", {"cwd": "/x", "model": "m", "context_window": 1}),
            ("turn_started", {"turn_id": str(uuid.uuid4())}),
            ("approval_decided", {"call_id": "missing", "approved": True}),
            ("tool_started", {"call_id": "missing"}),
            ("tool_finished", {"call_id": "missing", "status": "failed"}),
            ("tool_reconciled", {"call_id": "missing", "outcome": "failed"}),
            ("unknown_event", {}),
        ]
        for event_type, payload in invalid_cases:
            with self.subTest(event_type=event_type):
                event = Event.create(
                    seq=self.state.last_seq + 1,
                    session_id=self.session_id,
                    event_type=event_type,
                    payload=payload,
                )
                with self.assertRaises(DomainError):
                    SessionReducer.apply(self.state, event)
                self.assertEqual(self.state.last_seq, 2)

    def test_invalid_lifecycle_edges_are_rejected(self) -> None:
        self.start_turn()
        self.apply(
            "assistant_accepted",
            {"tool_calls": [self.tool("call-1", "write_file")]},
        )
        self.apply(
            "approval_decided",
            {"call_id": "call-1", "scope": "once", "approved": False},
        )

        start_denied = Event.create(
            seq=self.state.last_seq + 1,
            session_id=self.session_id,
            event_type="tool_started",
            payload={"call_id": "call-1"},
        )
        with self.assertRaises(DomainError):
            SessionReducer.apply(self.state, start_denied)

        open_turn_finish = Event.create(
            seq=self.state.last_seq + 1,
            session_id=self.session_id,
            event_type="turn_finished",
            payload={"turn_id": self.turn_id, "status": "completed"},
        )
        with self.assertRaises(DomainError):
            SessionReducer.apply(self.state, open_turn_finish)

        self.apply(
            "tool_finished",
            {"call_id": "call-1", "status": "denied", "result": "denied"},
        )
        duplicate_finish = Event.create(
            seq=self.state.last_seq + 1,
            session_id=self.session_id,
            event_type="tool_finished",
            payload={"call_id": "call-1", "status": "failed", "result": "again"},
        )
        with self.assertRaises(DomainError):
            SessionReducer.apply(self.state, duplicate_finish)

    def test_reducer_rejects_sequence_gaps_and_session_mismatches(self) -> None:
        self.create_session()
        gap = Event.create(
            seq=3,
            session_id=self.session_id,
            event_type="turn_started",
            payload={"turn_id": self.turn_id},
        )
        wrong_session = Event.create(
            seq=2,
            session_id=str(uuid.uuid4()),
            event_type="turn_started",
            payload={"turn_id": self.turn_id},
        )

        with self.assertRaisesRegex(DomainError, "sequence"):
            SessionReducer.apply(self.state, gap)
        with self.assertRaisesRegex(DomainError, "session"):
            SessionReducer.apply(self.state, wrong_session)


if __name__ == "__main__":
    unittest.main()
