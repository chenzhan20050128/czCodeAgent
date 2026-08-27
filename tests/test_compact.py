"""Compaction grouping, summarization, and durability contract tests."""

from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mca.compact import (
    SUMMARY_SECTIONS,
    CompactionError,
    SessionCompactor,
    atomic_groups,
    finalize_checkpoint,
    prepare_compaction,
)
from mca.domain import SamplingOutcome, SessionReducer, SessionState
from mca.model import SamplingResult
from mca.projection import (
    ProjectionEnvironment,
    PromptProjector,
    validate_conversation,
)
from mca.store import RolloutStore


def tool_call(call_id: str, name: str = "read_file") -> dict[str, object]:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": "{}"},
    }


class RecordingSummaryModel:
    def __init__(self, result: SamplingResult) -> None:
        self.result = result
        self.requests: list[
            tuple[list[Mapping[str, Any]], list[Mapping[str, Any]], bool]
        ] = []

    def sample(
        self,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]],
        allow_tools: bool,
        **_: object,
    ) -> SamplingResult:
        self.requests.append((list(messages), list(tools), allow_tools))
        return self.result


class AtomicCompactionTests(unittest.TestCase):
    def test_atomic_groups_never_split_an_assistant_from_all_tool_results(
        self,
    ) -> None:
        messages = [
            {"role": "system", "content": "live"},
            {"role": "user", "content": "original task"},
            {
                "role": "assistant",
                "content": None,
                "reasoning_content": "inspect both",
                "tool_calls": [tool_call("a"), tool_call("b")],
            },
            {"role": "tool", "tool_call_id": "a", "content": "A"},
            {"role": "tool", "tool_call_id": "b", "content": "B"},
            {"role": "assistant", "content": "plain answer"},
        ]

        groups = atomic_groups(messages)

        self.assertEqual([len(group.messages) for group in groups], [1, 3, 1])
        self.assertEqual(
            [message["role"] for message in groups[1].messages],
            ["assistant", "tool", "tool"],
        )
        for count in range(1, len(groups) + 1):
            tail = [
                message
                for group in groups[-count:]
                for message in group.messages
            ]
            validate_conversation(tail)

    def test_first_user_is_retained_once_when_it_overlaps_the_tail(self) -> None:
        messages = [
            {"role": "system", "content": "live"},
            {"role": "user", "content": "do the task"},
            {"role": "assistant", "content": "done"},
        ]

        draft = prepare_compaction(messages, through_seq=9, tail_group_count=2)
        payload = finalize_checkpoint(draft, "## Goal\nFinish the task.")

        replacement = payload["replacement_conversation"]
        self.assertEqual(
            sum(
                message == {"role": "user", "content": "do the task"}
                for message in replacement
            ),
            1,
        )
        self.assertEqual(replacement[-1], {"role": "assistant", "content": "done"})
        self.assertEqual(payload["through_seq"], 9)
        validate_conversation(replacement)

    def test_equal_user_text_from_different_turns_is_not_text_deduplicated(
        self,
    ) -> None:
        messages = [
            {"role": "system", "content": "live"},
            {"role": "user", "content": "same"},
            {"role": "assistant", "content": "first"},
            {"role": "user", "content": "same"},
            {"role": "assistant", "content": "second"},
        ]

        draft = prepare_compaction(messages, through_seq=10, tail_group_count=2)
        replacement = finalize_checkpoint(draft, "summary")[
            "replacement_conversation"
        ]

        self.assertEqual(
            [message for message in replacement if message["role"] == "user"],
            [
                {"role": "user", "content": "same"},
                {"role": "user", "content": "same"},
            ],
        )

    def test_oversized_summary_input_shortens_old_tools_deterministically(
        self,
    ) -> None:
        messages = [
            {"role": "system", "content": "live"},
            {"role": "user", "content": "task"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [tool_call("a"), tool_call("b")],
            },
            {
                "role": "tool",
                "tool_call_id": "a",
                "content": "A" * 500,
            },
            {
                "role": "tool",
                "tool_call_id": "b",
                "content": "B" * 500,
            },
            {"role": "assistant", "content": "recent"},
        ]

        first = prepare_compaction(
            messages,
            through_seq=8,
            tail_group_count=1,
            max_summary_input_chars=700,
            max_old_tool_chars=40,
        )
        second = prepare_compaction(
            messages,
            through_seq=8,
            tail_group_count=1,
            max_summary_input_chars=700,
            max_old_tool_chars=40,
        )

        self.assertEqual(first.summary_messages, second.summary_messages)
        transcript = first.summary_messages[1]["content"]
        self.assertIsInstance(transcript, str)
        serialized = transcript.split("\n", 1)[1]
        summary_conversation = json.loads(serialized)
        tools = [
            message
            for message in summary_conversation
            if message["role"] == "tool"
        ]
        self.assertEqual(
            [message["tool_call_id"] for message in tools], ["a", "b"]
        )
        self.assertTrue(
            all(
                "tool output shortened for compaction" in message["content"]
                for message in tools
            )
        )
        self.assertNotIn("A" * 100, serialized)
        replacement = finalize_checkpoint(first, "summary")[
            "replacement_conversation"
        ]
        self.assertEqual(replacement[-1], {"role": "assistant", "content": "recent"})


class SessionCompactorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.workspace = self.root / "work"
        self.workspace.mkdir()
        self.session_id = str(uuid.uuid4())
        self.turn_id = str(uuid.uuid4())
        self.store = RolloutStore.create(self.root / "sessions", self.session_id)
        self.addCleanup(self.store.close)
        self.state = SessionState()
        self.environment = ProjectionEnvironment(
            cwd=str(self.workspace),
            platform="test",
            date="2026-08-27",
            is_git=False,
        )
        self.append(
            "session_created",
            {
                "cwd": str(self.workspace),
                "model": "test-model",
                "context_window": 4096,
            },
        )
        self.append(
            "turn_started",
            {"turn_id": self.turn_id, "user_input": "inspect the project"},
        )

    def append(self, event_type: str, payload: dict[str, object]) -> None:
        event = self.store.append(event_type, payload)
        SessionReducer.apply(self.state, event)

    def accept_plain_assistant(self) -> None:
        self.append(
            "assistant_accepted",
            {
                "turn_id": self.turn_id,
                "content": "working context",
                "reasoning_content": "private tail reasoning",
                "tool_calls": [],
            },
        )

    def make_compactor(self, model: RecordingSummaryModel, **kwargs: object) -> SessionCompactor:
        return SessionCompactor(
            store=self.store,
            state=self.state,
            model=model,
            environment=lambda: self.environment,
            **kwargs,
        )

    def test_summary_sample_has_fixed_sections_and_never_exposes_tools(self) -> None:
        self.accept_plain_assistant()
        model = RecordingSummaryModel(
            SamplingResult(
                SamplingOutcome.COMPLETE_TEXT,
                content="## Goal\nContinue safely.",
                reasoning_content="do not persist this",
                finish_reason="stop",
            )
        )

        self.assertTrue(self.make_compactor(model)())

        self.assertEqual(len(model.requests), 1)
        messages, tools, allow_tools = model.requests[0]
        self.assertEqual(tools, [])
        self.assertIs(allow_tools, False)
        prompt = messages[0]["content"]
        for section in SUMMARY_SECTIONS:
            self.assertIn(section, prompt)
        checkpoint = self.store.load()[-1]
        self.assertEqual(checkpoint.type, "compaction_checkpoint")
        self.assertEqual(checkpoint.payload["through_seq"], checkpoint.seq - 1)
        self.assertEqual(checkpoint.payload["summary"], "## Goal\nContinue safely.")
        replacement = checkpoint.to_dict()["payload"][
            "replacement_conversation"
        ]
        validate_conversation(replacement)
        self.assertEqual(
            replacement[-1]["reasoning_content"], "private tail reasoning"
        )
        self.assertNotIn("do not persist this", json.dumps(replacement))
        projected = PromptProjector.project(
            self.store.load(), self.state, self.environment
        )
        validate_conversation(projected)

    def test_invalid_or_empty_summary_does_not_append_a_checkpoint(self) -> None:
        invalid_results = (
            SamplingResult(SamplingOutcome.COMPLETE_TEXT, content=""),
            SamplingResult(
                SamplingOutcome.VALID_TOOL_BATCH,
                tool_calls=(),
                finish_reason="tool_calls",
            ),
            SamplingResult(SamplingOutcome.PROTOCOL_ERROR, error="bad"),
        )
        for result in invalid_results:
            with self.subTest(result=result.outcome):
                fresh = SessionCompactorTests(methodName="runTest")
                fresh.setUp()
                self.addCleanup(fresh.doCleanups)
                fresh.accept_plain_assistant()
                before = fresh.store.load()
                model = RecordingSummaryModel(result)

                with self.assertRaises(CompactionError):
                    fresh.make_compactor(model).compact()

                self.assertEqual(fresh.store.load(), before)
                self.assertIsNone(fresh.state.latest_checkpoint)

    def test_checkpoint_append_failure_leaves_the_old_state_and_view_active(self) -> None:
        self.accept_plain_assistant()
        model = RecordingSummaryModel(
            SamplingResult(SamplingOutcome.COMPLETE_TEXT, content="valid summary")
        )
        before_events = self.store.load()
        before_state = SessionReducer.replay(before_events)
        before_projection = PromptProjector.project(
            before_events, self.state, self.environment
        )

        with patch.object(self.store, "append", side_effect=OSError("disk full")):
            with self.assertRaisesRegex(CompactionError, "append"):
                self.make_compactor(model).compact()

        self.assertEqual(self.state, before_state)
        self.assertEqual(self.store.load(), before_events)
        self.assertEqual(
            PromptProjector.project(self.store.load(), self.state, self.environment),
            before_projection,
        )

    def test_new_turn_without_assistant_and_idle_session_can_compact(self) -> None:
        model = RecordingSummaryModel(
            SamplingResult(SamplingOutcome.COMPLETE_TEXT, content="summary")
        )

        self.assertTrue(self.make_compactor(model).compact())
        first_checkpoint = self.store.load()[-1]
        first_replacement = list(
            first_checkpoint.payload["replacement_conversation"]
        )
        self.assertEqual(
            first_replacement,
            [{"role": "user", "content": "inspect the project"}],
        )

        self.append(
            "turn_finished",
            {"turn_id": self.turn_id, "status": "completed"},
        )
        self.assertTrue(self.make_compactor(model).compact())
        self.assertIsNone(self.state.active_turn_id)
        self.assertEqual(len(model.requests), 2)

    def test_unresolved_tool_batch_fails_before_the_summary_sample(self) -> None:
        model = RecordingSummaryModel(
            SamplingResult(SamplingOutcome.COMPLETE_TEXT, content="summary")
        )

        self.append(
            "assistant_accepted",
            {
                "turn_id": self.turn_id,
                "content": None,
                "tool_calls": [tool_call("open")],
            },
        )
        with self.assertRaisesRegex(CompactionError, "sampling boundary"):
            self.make_compactor(model).compact()
        self.assertEqual(model.requests, [])


if __name__ == "__main__":
    unittest.main()
