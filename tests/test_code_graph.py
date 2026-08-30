"""Durable Code Mode graph domain and projection tests."""

from __future__ import annotations

import sys
import unittest
import uuid
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mca.code_graph import CodeNodeStatus, CodeRunStatus, graph_summary
from mca.domain import DomainError, Event, SessionReducer, SessionState, ToolStatus
from mca.projection import ProjectionEnvironment, PromptProjector


class CodeGraphDomainTests(unittest.TestCase):
    def setUp(self) -> None:
        self.session_id = str(uuid.uuid4())
        self.turn_id = str(uuid.uuid4())
        self.run_id = str(uuid.uuid4())
        self.state = SessionState()
        self.events: list[Event] = []
        self.apply(
            "session_created",
            {"cwd": "/workspace", "model": "m", "context_window": 4096},
        )
        self.apply(
            "turn_started",
            {"turn_id": self.turn_id, "user_input": "run code"},
        )
        self.apply(
            "assistant_accepted",
            {
                "content": None,
                "tool_calls": [
                    {
                        "id": "outer",
                        "type": "function",
                        "function": {
                            "name": "run_code",
                            "arguments": '{"description":"work","code":"return 1"}',
                        },
                    }
                ],
            },
        )
        self.outer_key = "3:outer"
        self.apply("tool_started", {"call_key": self.outer_key, "call_id": "outer"})
        self.apply(
            "code_run_started",
            {
                "run_id": self.run_id,
                "turn_id": self.turn_id,
                "parent_call_key": self.outer_key,
                "description": "work",
                "source_hash": "sha256:program",
            },
        )

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

    def plan(
        self, ordinal: int, name: str, dependencies: list[str] | None = None
    ) -> str:
        node_id = f"{self.run_id}:node:{ordinal}"
        self.apply(
            "code_node_planned",
            {
                "run_id": self.run_id,
                "node_id": node_id,
                "ordinal": ordinal,
                "name": name,
                "arguments": "{}",
                "dependencies": dependencies or [],
            },
        )
        return node_id

    def test_planned_nodes_register_internal_calls_and_dependencies(self) -> None:
        first = self.plan(1, "write_file")
        second = self.plan(2, "bash", [first])

        run = self.state.code_runs[self.run_id]
        self.assertIs(run.status, CodeRunStatus.ACTIVE)
        self.assertEqual(run.node_ids, (first, second))
        self.assertEqual(self.state.code_nodes[second].dependencies, (first,))
        self.assertIs(self.state.code_nodes[second].status, CodeNodeStatus.PLANNED)
        nested = self.state.tool_calls[second]
        self.assertEqual(nested.origin, "code")
        self.assertEqual(nested.parent_call_key, self.outer_key)
        self.assertEqual(nested.code_run_id, self.run_id)
        self.assertEqual(nested.ordinal, 2)

    def test_planning_rejects_unknown_dependency_duplicate_ordinal_and_cycle(self) -> None:
        first = self.plan(1, "read_file")
        cases = (
            {
                "run_id": self.run_id,
                "node_id": f"{self.run_id}:node:2",
                "ordinal": 2,
                "name": "bash",
                "arguments": "{}",
                "dependencies": ["missing"],
            },
            {
                "run_id": self.run_id,
                "node_id": f"{self.run_id}:node:other",
                "ordinal": 1,
                "name": "bash",
                "arguments": "{}",
                "dependencies": [first],
            },
            {
                "run_id": self.run_id,
                "node_id": f"{self.run_id}:node:3",
                "ordinal": 3,
                "name": "bash",
                "arguments": "{}",
                "dependencies": [f"{self.run_id}:node:3"],
            },
        )
        for payload in cases:
            with self.subTest(payload=payload):
                candidate = Event.create(
                    seq=self.state.last_seq + 1,
                    session_id=self.session_id,
                    event_type="code_node_planned",
                    payload=payload,
                )
                with self.assertRaises(DomainError):
                    SessionReducer.apply(self.state, candidate)

    def test_nested_tool_lifecycle_updates_node_and_summary(self) -> None:
        first = self.plan(1, "read_file")
        second = self.plan(2, "bash", [first])
        self.apply(
            "tool_started",
            {"call_key": first, "call_id": first, "origin": "code"},
        )
        self.apply(
            "tool_finished",
            {
                "call_key": first,
                "call_id": first,
                "origin": "code",
                "status": "failed",
                "result": "read failed",
            },
        )
        self.apply(
            "tool_finished",
            {
                "call_key": second,
                "call_id": second,
                "origin": "code",
                "status": "upstream_failed",
                "result": "dependency failed",
                "blocked_by": [first],
                "root_failures": [first],
            },
        )

        self.assertIs(self.state.code_nodes[first].status, CodeNodeStatus.FAILED)
        self.assertIs(
            self.state.code_nodes[second].status, CodeNodeStatus.UPSTREAM_FAILED
        )
        summary = graph_summary(self.state, self.run_id)
        self.assertEqual(summary["failed"], 1)
        self.assertEqual(summary["upstream_failed"], 1)
        self.assertEqual(summary["root_failures"], [first])

    def test_code_run_finishes_only_after_all_nodes_and_replays_exactly(self) -> None:
        node = self.plan(1, "read_file")
        self.apply(
            "tool_finished",
            {
                "call_key": node,
                "call_id": node,
                "origin": "code",
                "status": "not_executed",
                "result": "not run",
            },
        )
        self.apply(
            "code_run_finished",
            {
                "run_id": self.run_id,
                "status": "failed",
                "result": "runtime stopped",
                "summary": graph_summary(self.state, self.run_id),
            },
        )

        self.assertIs(
            self.state.code_runs[self.run_id].status, CodeRunStatus.FAILED
        )
        self.assertEqual(SessionReducer.replay(self.events), self.state)

    def test_nested_results_are_hidden_from_provider_projection(self) -> None:
        node = self.plan(1, "read_file")
        self.apply(
            "tool_started",
            {"call_key": node, "call_id": node, "origin": "code"},
        )
        self.apply(
            "tool_finished",
            {
                "call_key": node,
                "call_id": node,
                "origin": "code",
                "status": "succeeded",
                "result": "large nested output",
            },
        )
        self.apply(
            "code_run_finished",
            {
                "run_id": self.run_id,
                "status": "succeeded",
                "result": "curated",
                "summary": graph_summary(self.state, self.run_id),
            },
        )
        self.apply(
            "tool_finished",
            {
                "call_key": self.outer_key,
                "call_id": "outer",
                "status": "succeeded",
                "result": "curated",
            },
        )

        messages = PromptProjector.project(
            self.events,
            self.state,
            ProjectionEnvironment(
                cwd="/workspace", platform="test", date="2026-08-30", is_git=False
            ),
        )
        tool_messages = [message for message in messages if message["role"] == "tool"]
        self.assertEqual(tool_messages, [
            {"role": "tool", "tool_call_id": "outer", "content": "curated"}
        ])

    def test_reconciliation_updates_nested_node_status(self) -> None:
        node = self.plan(1, "write_file")
        self.apply(
            "tool_started",
            {"call_key": node, "call_id": node, "origin": "code"},
        )
        self.apply(
            "tool_finished",
            {
                "call_key": node,
                "call_id": node,
                "origin": "code",
                "status": "outcome_unknown",
                "result": "unknown",
                "recovery_blocked": True,
            },
        )
        self.apply(
            "tool_reconciled",
            {
                "call_key": node,
                "call_id": node,
                "origin": "code",
                "outcome": "succeeded",
                "note": "verified",
            },
        )

        self.assertIs(
            self.state.code_nodes[node].status,
            CodeNodeStatus.USER_CONFIRMED_SUCCESS,
        )


if __name__ == "__main__":
    unittest.main()
