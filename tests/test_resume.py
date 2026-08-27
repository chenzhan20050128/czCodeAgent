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

from mca.domain import (
    DomainError,
    Event,
    SessionReducer,
    SessionState,
    ToolStatus,
    TurnStatus,
)
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


class SimulatedCrash(BaseException):
    """Test-only process interruption after one durable append."""


class ResumeTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.workspace = (self.root / "work").resolve()
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

    def create_rollout(
        self,
        name: str,
        call_ids: tuple[str, ...],
        *,
        started: bool = False,
    ) -> tuple[Path, Path, str, str]:
        root = self.root / name
        workspace = (root / "work").resolve()
        workspace.mkdir(parents=True)
        sessions = root / "sessions"
        session_id = str(uuid.uuid4())
        turn_id = str(uuid.uuid4())
        with RolloutStore.create(sessions, session_id) as store:
            store.append(
                "session_created",
                {
                    "cwd": str(workspace),
                    "model": "test-model",
                    "context_window": 4096,
                },
            )
            store.append(
                "turn_started",
                {"turn_id": turn_id, "user_input": "recover this task"},
            )
            store.append(
                "assistant_accepted",
                {
                    "content": None,
                    "tool_calls": [tool_call(call_id) for call_id in call_ids],
                },
            )
            if started:
                for call_id in call_ids:
                    store.append(
                        "tool_started",
                        {
                            "call_key": f"3:{call_id}",
                            "call_id": call_id,
                        },
                    )
        return sessions, workspace, session_id, turn_id


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
            self.assertEqual(
                resumed.store.load()[-3].type, "turn_recovery_intent"
            )
            self.assertEqual(
                resumed.store.load()[-3].payload["action"],
                "recover_interrupted",
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

    def test_requested_recovery_completes_after_every_durable_append_crash(
        self,
    ) -> None:
        expected_types = [
            "turn_recovery_intent",
            "tool_finished",
            "tool_finished",
            "turn_finished",
        ]
        original_append = RolloutStore.append

        for crash_at in range(1, len(expected_types) + 1):
            with self.subTest(crash_at=crash_at):
                sessions, workspace, session_id, turn_id = self.create_rollout(
                    f"requested-crash-{crash_at}", ("a", "b")
                )
                append_count = 0

                def append_then_crash(
                    store: RolloutStore,
                    event_or_type: Event | str,
                    payload: dict[str, object] | None = None,
                ) -> Event:
                    nonlocal append_count
                    event = original_append(store, event_or_type, payload)
                    if store.session_id == session_id:
                        append_count += 1
                        if append_count == crash_at:
                            raise SimulatedCrash(
                                f"crash after durable append {crash_at}"
                            )
                    return event

                with patch.object(RolloutStore, "append", new=append_then_crash):
                    with self.assertRaises(SimulatedCrash):
                        resume_session(sessions, session_id, workspace)

                with resume_session(sessions, session_id, workspace) as resumed:
                    events = resumed.store.load()
                    recovery = events[3:]
                    self.assertEqual(
                        [event.type for event in recovery], expected_types
                    )
                    self.assertEqual(
                        [
                            event.payload["status"]
                            for event in recovery
                            if event.type == "tool_finished"
                        ],
                        ["not_executed", "not_executed"],
                    )
                    self.assertEqual(
                        sum(event.type == "turn_finished" for event in events),
                        1,
                    )
                    self.assertIs(
                        resumed.state.turns[turn_id], TurnStatus.INTERRUPTED
                    )
                    self.assertIsNone(resumed.state.pending_recovery_intent)
                    self.assertIsNone(continuable_turn_id(resumed.state))
                    self.assertEqual(
                        resumed.state, SessionReducer.replay(events)
                    )
                    messages = PromptProjector.project(
                        events,
                        resumed.state,
                        ProjectionEnvironment(
                            cwd=str(workspace),
                            platform="test",
                            date="2026-08-27",
                            is_git=False,
                        ),
                    )
                    validate_conversation(messages)
                    self.assertEqual(
                        [
                            message["tool_call_id"]
                            for message in messages
                            if message["role"] == "tool"
                        ],
                        ["a", "b"],
                    )

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
            self.assertIsNone(resumed.state.pending_recovery_intent)
            self.assertNotIn(
                "turn_recovery_intent",
                [event.type for event in resumed.store.load()],
            )
            with self.assertRaisesRegex(ReconciliationError, "blocked"):
                continuable_turn_id(resumed.state)
            with self.assertRaises(ProjectionBlockedError):
                PromptProjector.project(
                    resumed.store.load(), resumed.state, self.environment()
                )

    def test_mixed_started_replay_never_creates_interrupted_intent(self) -> None:
        self.start_turn(("started", "pending"))
        self.append(
            "tool_started",
            {"call_key": "3:started", "call_id": "started"},
        )
        self.store.close()
        original_append = RolloutStore.append

        def append_then_crash(
            store: RolloutStore,
            event_or_type: Event | str,
            payload: dict[str, object] | None = None,
        ) -> Event:
            event = original_append(store, event_or_type, payload)
            raise SimulatedCrash("crash after outcome_unknown was durable")

        with patch.object(RolloutStore, "append", new=append_then_crash):
            with self.assertRaises(SimulatedCrash):
                resume_session(self.sessions, self.session_id, self.workspace)

        with resume_session(
            self.sessions, self.session_id, self.workspace
        ) as resumed:
            self.assertIs(
                resumed.state.tool_calls["3:started"].status,
                ToolStatus.OUTCOME_UNKNOWN,
            )
            self.assertIs(
                resumed.state.tool_calls["3:pending"].status,
                ToolStatus.NOT_EXECUTED,
            )
            self.assertTrue(resumed.state.recovery_blocked)
            self.assertIsNone(resumed.state.pending_recovery_intent)
            self.assertNotIn(
                "turn_recovery_intent",
                [event.type for event in resumed.store.load()],
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

    def test_resume_rejects_recorded_symlink_even_after_it_is_redirected(
        self,
    ) -> None:
        first = self.root / "first"
        second = self.root / "second"
        first.mkdir()
        second.mkdir()
        link = self.root / "workspace-link"
        link.symlink_to(first, target_is_directory=True)
        session_id = str(uuid.uuid4())
        with RolloutStore.create(self.root / "linked-sessions", session_id) as store:
            store.append(
                "session_created",
                {
                    "cwd": str(link),
                    "model": "test-model",
                    "context_window": 4096,
                },
            )

        link.unlink()
        link.symlink_to(second, target_is_directory=True)

        with self.assertRaisesRegex(ResumeError, "recorded cwd.*canonical"):
            resume_session(self.root / "linked-sessions", session_id, link)

    def test_recovery_intent_schema_and_transitions_are_strict(self) -> None:
        self.start_turn(("pending",))
        valid_payload = {
            "turn_id": self.turn_id,
            "action": "recover_interrupted",
            "reason": "resume found an unstarted call",
        }
        invalid_payloads = (
            {"turn_id": self.turn_id, "action": "recover_interrupted"},
            {**valid_payload, "extra": True},
            {**valid_payload, "turn_id": str(uuid.uuid4())},
            {**valid_payload, "action": "retry"},
            {**valid_payload, "reason": None},
        )
        for payload in invalid_payloads:
            with self.subTest(payload=payload):
                state = SessionReducer.replay(self.state.events)
                event = Event.create(
                    seq=state.last_seq + 1,
                    session_id=self.session_id,
                    event_type="turn_recovery_intent",
                    payload=payload,
                )
                with self.assertRaises(DomainError):
                    SessionReducer.apply(state, event)

        state = SessionReducer.replay(self.state.events)
        intent = Event.create(
            seq=state.last_seq + 1,
            session_id=self.session_id,
            event_type="turn_recovery_intent",
            payload=valid_payload,
        )
        SessionReducer.apply(state, intent)
        self.assertEqual(state.pending_recovery_intent, intent)
        for action in ("recover_interrupted", "abandon"):
            duplicate = Event.create(
                seq=state.last_seq + 1,
                session_id=self.session_id,
                event_type="turn_recovery_intent",
                payload={**valid_payload, "action": action},
            )
            with self.subTest(second_action=action):
                with self.assertRaisesRegex(DomainError, "already pending"):
                    SessionReducer.apply(state, duplicate)

        terminal_state = SessionReducer.replay(self.state.events[:1])
        turn_started = Event.create(
            seq=terminal_state.last_seq + 1,
            session_id=self.session_id,
            event_type="turn_started",
            payload={"turn_id": str(uuid.uuid4()), "user_input": "done"},
        )
        SessionReducer.apply(terminal_state, turn_started)
        turn_finished = Event.create(
            seq=terminal_state.last_seq + 1,
            session_id=self.session_id,
            event_type="turn_finished",
            payload={
                "turn_id": turn_started.payload["turn_id"],
                "status": "completed",
            },
        )
        SessionReducer.apply(terminal_state, turn_finished)
        after_terminal = Event.create(
            seq=terminal_state.last_seq + 1,
            session_id=self.session_id,
            event_type="turn_recovery_intent",
            payload={
                "turn_id": turn_started.payload["turn_id"],
                "action": "recover_interrupted",
                "reason": "too late",
            },
        )
        with self.assertRaisesRegex(DomainError, "active turn"):
            SessionReducer.apply(terminal_state, after_terminal)

    def test_pending_intent_blocks_projection_until_matching_finish(self) -> None:
        self.start_turn(("pending",))
        self.append(
            "turn_recovery_intent",
            {
                "turn_id": self.turn_id,
                "action": "recover_interrupted",
                "reason": "resume found an unstarted call",
            },
        )
        self.append(
            "tool_finished",
            {
                "call_key": "3:pending",
                "call_id": "pending",
                "status": "not_executed",
                "result": "tool call was not started before recovery",
                "recovery_blocked": False,
            },
        )
        with self.assertRaises(ProjectionBlockedError):
            PromptProjector.project(
                self.store.load(), self.state, self.environment()
            )
        with self.assertRaisesRegex(ReconciliationError, "recovery intent"):
            continuable_turn_id(self.state)

        mismatched = Event.create(
            seq=self.state.last_seq + 1,
            session_id=self.session_id,
            event_type="turn_finished",
            payload={
                "turn_id": self.turn_id,
                "status": "abandoned",
            },
        )
        with self.assertRaisesRegex(DomainError, "recovery intent"):
            SessionReducer.apply(self.state, mismatched)

        self.append(
            "turn_finished",
            {
                "turn_id": self.turn_id,
                "status": "interrupted",
            },
        )
        self.assertIsNone(self.state.pending_recovery_intent)
        messages = PromptProjector.project(
            self.store.load(), self.state, self.environment()
        )
        validate_conversation(messages)
        self.assertEqual(
            [message["tool_call_id"] for message in messages if message["role"] == "tool"],
            ["pending"],
        )

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
            self.assertEqual(
                [event.type for event in resumed.store.load()[-4:]],
                [
                    "turn_recovery_intent",
                    "tool_reconciled",
                    "tool_reconciled",
                    "turn_finished",
                ],
            )
            self.assertIsNone(resumed.state.pending_recovery_intent)
            self.assertIsNone(continuable_turn_id(resumed.state))

    def test_abandon_completes_after_every_durable_append_crash(self) -> None:
        expected_types = [
            "turn_recovery_intent",
            "tool_reconciled",
            "tool_reconciled",
            "turn_finished",
        ]
        for crash_at in range(1, len(expected_types) + 1):
            with self.subTest(crash_at=crash_at):
                sessions, workspace, session_id, turn_id = self.create_rollout(
                    f"abandon-crash-{crash_at}", ("a", "b"), started=True
                )
                with resume_session(sessions, session_id, workspace):
                    pass
                resumed = resume_session(sessions, session_id, workspace)
                before = len(resumed.store.load())
                original_append = resumed.store.append
                append_count = 0

                def append_then_crash(
                    event_or_type: Event | str,
                    payload: dict[str, object] | None = None,
                ) -> Event:
                    nonlocal append_count
                    event = original_append(event_or_type, payload)
                    append_count += 1
                    if append_count == crash_at:
                        raise SimulatedCrash(
                            f"crash after durable append {crash_at}"
                        )
                    return event

                try:
                    with patch.object(
                        resumed.store, "append", side_effect=append_then_crash
                    ):
                        with self.assertRaises(SimulatedCrash):
                            reconcile_tool(
                                resumed.store,
                                resumed.state,
                                "3:a",
                                "abandoned",
                                "cannot verify safely",
                            )
                finally:
                    resumed.close()

                with resume_session(sessions, session_id, workspace) as recovered:
                    events = recovered.store.load()
                    recovery = events[before:]
                    self.assertEqual(
                        [event.type for event in recovery], expected_types
                    )
                    self.assertEqual(
                        [
                            event.payload["call_id"]
                            for event in recovery
                            if event.type == "tool_reconciled"
                        ],
                        ["a", "b"],
                    )
                    self.assertTrue(
                        all(
                            event.payload["note"] == "cannot verify safely"
                            for event in recovery
                            if event.type == "tool_reconciled"
                        )
                    )
                    self.assertEqual(
                        sum(event.type == "turn_finished" for event in events),
                        1,
                    )
                    self.assertIs(
                        recovered.state.turns[turn_id], TurnStatus.ABANDONED
                    )
                    self.assertIsNone(recovered.state.pending_recovery_intent)
                    self.assertIsNone(continuable_turn_id(recovered.state))
                    self.assertEqual(
                        recovered.state, SessionReducer.replay(events)
                    )
                    messages = PromptProjector.project(
                        events,
                        recovered.state,
                        ProjectionEnvironment(
                            cwd=str(workspace),
                            platform="test",
                            date="2026-08-27",
                            is_git=False,
                        ),
                    )
                    validate_conversation(messages)
                    self.assertEqual(
                        [
                            message["tool_call_id"]
                            for message in messages
                            if message["role"] == "tool"
                        ],
                        ["a", "b"],
                    )

    def test_apply_interruption_closes_store_and_replay_finishes_abandon(
        self,
    ) -> None:
        with self.resume_unknowns("a", "b") as resumed:
            original_apply = SessionReducer.apply
            intent_apply_count = 0

            def apply_then_crash(state: SessionState, event: Event) -> SessionState:
                nonlocal intent_apply_count
                result = original_apply(state, event)
                if event.type == "turn_recovery_intent":
                    intent_apply_count += 1
                    if intent_apply_count == 2:
                        raise SimulatedCrash("crash after applying durable intent")
                return result

            with patch.object(
                SessionReducer, "apply", side_effect=apply_then_crash
            ):
                with self.assertRaises(SimulatedCrash):
                    reconcile_tool(
                        resumed.store,
                        resumed.state,
                        "3:a",
                        "abandoned",
                        "cannot verify safely",
                    )
            with self.assertRaisesRegex(ValueError, "closed"):
                resumed.store.load()

        with self.reopen() as recovered:
            self.assertIs(
                recovered.state.turns[self.turn_id], TurnStatus.ABANDONED
            )
            self.assertIsNone(recovered.state.pending_recovery_intent)
            self.assertEqual(
                recovered.state, SessionReducer.replay(recovered.store.load())
            )

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
