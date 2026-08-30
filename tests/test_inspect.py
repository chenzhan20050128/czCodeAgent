"""Tests for read-only session inspection projections."""

from __future__ import annotations

import json
import sys
import uuid
from pathlib import Path
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mca.code_graph import graph_summary
from mca.domain import Event, SessionReducer, SessionState
from mca.inspect import (
    SessionSummary,
    list_session_ids,
    render_transcript,
    summarize,
)


class InspectTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.session_id = str(uuid.uuid4())
        self.turn_id = str(uuid.uuid4())
        self.events: list[Event] = []
        self.state = SessionState()

    def apply(self, event_type: str, payload: dict[str, object]) -> Event:
        event = Event.create(
            seq=len(self.events) + 1,
            session_id=self.session_id,
            event_type=event_type,
            payload=payload,
        )
        SessionReducer.apply(self.state, event)
        self.events.append(event)
        return event

    def create_session(self) -> None:
        self.apply(
            "session_created",
            {"cwd": "/work", "model": "test-model", "context_window": 4096},
        )

    def tool(self, call_id: str, name: str, arguments: str = "{}") -> dict[str, object]:
        return {
            "id": call_id,
            "type": "function",
            "function": {"name": name, "arguments": arguments},
        }


class SummarizeTests(InspectTestCase):
    def test_summarize_empty_state_reports_no_session(self) -> None:
        summary = summarize(SessionState())
        self.assertIsInstance(summary, SessionSummary)
        self.assertIsNone(summary.session_id)
        self.assertEqual(summary.turn_count, 0)
        self.assertEqual(summary.tool_call_count, 0)

    def test_summarize_reports_model_turns_and_last_status(self) -> None:
        self.create_session()
        self.apply("turn_started", {"turn_id": self.turn_id, "user_input": "do it"})
        self.apply(
            "assistant_accepted",
            {"turn_id": self.turn_id, "content": "hi", "tool_calls": []},
        )
        self.apply(
            "turn_finished",
            {"turn_id": self.turn_id, "status": "completed", "final_text": "hi"},
        )

        summary = summarize(self.state)

        self.assertEqual(summary.session_id, self.session_id)
        self.assertEqual(summary.model, "test-model")
        self.assertEqual(summary.turn_count, 1)
        self.assertEqual(summary.last_turn_status, "completed")
        self.assertFalse(summary.is_active)
        self.assertFalse(summary.recovery_blocked)

    def test_summarize_counts_tool_calls_and_flags_active_turn(self) -> None:
        self.create_session()
        self.apply("turn_started", {"turn_id": self.turn_id, "user_input": "x"})
        self.apply(
            "assistant_accepted",
            {"turn_id": self.turn_id, "tool_calls": [self.tool("c1", "read_file")]},
        )

        summary = summarize(self.state)

        self.assertEqual(summary.tool_call_count, 1)
        self.assertTrue(summary.is_active)
        self.assertEqual(summary.last_turn_status, "active")


class RenderTranscriptTests(InspectTestCase):
    def build_finished_code_run(self) -> str:
        self.create_session()
        self.apply("turn_started", {"turn_id": self.turn_id, "user_input": "run graph"})
        self.apply("assistant_accepted", {"content": None, "tool_calls": [self.tool("outer", "run_code")]})
        outer_key = "3:outer"
        self.apply("tool_started", {"call_key": outer_key, "call_id": "outer"})
        run_id = str(uuid.uuid4())
        self.apply("code_run_started", {"run_id": run_id, "turn_id": self.turn_id, "parent_call_key": outer_key, "description": "inspect and test", "source_hash": "sha256:x"})
        first = f"{run_id}:node:1"
        second = f"{run_id}:node:2"
        self.apply("code_node_planned", {"run_id": run_id, "node_id": first, "ordinal": 1, "name": "read_file", "arguments": '{"path":"src/a.py"}', "dependencies": []})
        self.apply("code_node_planned", {"run_id": run_id, "node_id": second, "ordinal": 2, "name": "bash", "arguments": '{"command":"python3 -m unittest"}', "dependencies": [first]})
        self.apply("tool_started", {"call_key": first, "call_id": first, "origin": "code"})
        self.apply("tool_finished", {"call_key": first, "call_id": first, "origin": "code", "status": "succeeded", "result": "nested read output"})
        self.apply("approval_decided", {"call_key": second, "call_id": second, "approved": True, "scope": "once"})
        self.apply("tool_started", {"call_key": second, "call_id": second, "origin": "code"})
        self.apply("tool_finished", {"call_key": second, "call_id": second, "origin": "code", "status": "failed", "result": "tests failed"})
        self.apply("code_run_finished", {"run_id": run_id, "status": "failed", "result": "summary", "summary": graph_summary(self.state, run_id)})
        self.apply("tool_finished", {"call_key": outer_key, "call_id": "outer", "status": "failed", "result": "outer summary"})
        return run_id

    def test_transcript_of_empty_state_is_explicit(self) -> None:
        text = render_transcript(SessionState())
        self.assertIn("no session", text.lower())

    def test_transcript_includes_user_assistant_and_turn_status(self) -> None:
        self.create_session()
        self.apply(
            "turn_started",
            {"turn_id": self.turn_id, "user_input": "fix the bug"},
        )
        self.apply(
            "assistant_accepted",
            {"turn_id": self.turn_id, "content": "on it", "tool_calls": []},
        )
        self.apply(
            "turn_finished",
            {"turn_id": self.turn_id, "status": "completed", "final_text": "on it"},
        )

        text = render_transcript(self.state)

        self.assertIn("fix the bug", text)
        self.assertIn("on it", text)
        self.assertIn("completed", text)
        self.assertIn(self.session_id, text)

    def test_transcript_renders_tool_calls_with_terminal_status(self) -> None:
        self.create_session()
        self.apply("turn_started", {"turn_id": self.turn_id, "user_input": "run"})
        self.apply(
            "assistant_accepted",
            {
                "turn_id": self.turn_id,
                "tool_calls": [self.tool("c1", "bash", '{"command":"ls"}')],
            },
        )
        self.apply("approval_decided", {"call_id": "c1", "approved": True})
        self.apply("tool_started", {"call_id": "c1"})
        self.apply(
            "tool_finished",
            {"call_id": "c1", "status": "succeeded", "result": "out"},
        )

        text = render_transcript(self.state)

        self.assertIn("bash", text)
        self.assertIn("succeeded", text)

    def test_transcript_truncates_a_very_long_tool_result(self) -> None:
        self.create_session()
        self.apply("turn_started", {"turn_id": self.turn_id, "user_input": "run"})
        self.apply(
            "assistant_accepted",
            {"turn_id": self.turn_id, "tool_calls": [self.tool("c1", "bash", "{}")]},
        )
        self.apply("approval_decided", {"call_id": "c1", "approved": True})
        self.apply("tool_started", {"call_id": "c1"})
        self.apply(
            "tool_finished",
            {"call_id": "c1", "status": "succeeded", "result": "x" * 5000},
        )

        text = render_transcript(self.state)

        self.assertLess(len(text), 5000)
        self.assertIn("truncated", text.lower())

    def test_transcript_embeds_compact_code_graph_and_hides_nested_tool_noise(self) -> None:
        self.build_finished_code_run()

        text = render_transcript(self.state)

        self.assertIn("run_code: inspect and test FAILED", text)
        self.assertIn("#1 ✓ SUCCEEDED  read_file", text)
        self.assertIn("#2 ✗ FAILED  APPROVED  bash", text)
        self.assertNotIn("#1 ──▶ #2", text)
        self.assertNotIn("nested read output", text)

    def test_expanded_transcript_reconstructs_dependency_edges_from_replay(self) -> None:
        self.build_finished_code_run()
        replayed = SessionReducer.replay(self.events)

        live = render_transcript(self.state, expanded_graph=True)
        cold = render_transcript(replayed, expanded_graph=True)

        self.assertEqual(cold, live)
        self.assertIn("#1 ──▶ #2", cold)
        self.assertIn("tests failed", cold)


class ListSessionIdsTests(unittest.TestCase):
    def test_list_returns_sorted_valid_session_ids_only(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "sessions"
            root.mkdir()
            ids = sorted(str(uuid.uuid4()) for _ in range(3))
            for session_id in ids:
                (root / f"{session_id}.jsonl").write_text("", encoding="utf-8")
            (root / "not-a-session.jsonl").write_text("", encoding="utf-8")
            (root / f"{uuid.uuid4()}.txt").write_text("", encoding="utf-8")

            self.assertEqual(list_session_ids(root), ids)

    def test_list_of_missing_root_is_empty(self) -> None:
        self.assertEqual(list_session_ids(Path("/nonexistent/sessions")), [])


if __name__ == "__main__":
    unittest.main()
