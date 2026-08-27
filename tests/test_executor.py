"""Tool execution pipeline and durable event ordering tests."""

from __future__ import annotations

import base64
import hashlib
import sys
import tempfile
import unittest
import uuid
from pathlib import Path
from unittest.mock import Mock, patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mca.approval import ApprovalDecision
from mca.domain import SessionReducer, SessionState, ToolStatus
from mca.executor import AcceptedToolCall, ToolExecutor, ToolExecutorError
from mca.store import RolloutStore
from mca.tools import create_tool_registry
from mca.tools.filesystem import PreparedFileChange
from mca.tools.registry import SideEffect, ToolRegistry, ToolResult, ToolSpec


class RecordingApprover:
    def __init__(self, decision: ApprovalDecision) -> None:
        self.decision = decision
        self.requests: list[object] = []

    def decide(self, request: object) -> ApprovalDecision:
        self.requests.append(request)
        return self.decision


class ExecutorTestCase(unittest.TestCase):
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
        self.append(
            "session_created",
            {
                "cwd": str(self.workspace),
                "model": "test-model",
                "context_window": 4096,
            },
        )
        self.append(
            "turn_started", {"turn_id": self.turn_id, "user_input": "test"}
        )

    def append(self, event_type: str, payload: dict[str, object]) -> None:
        event = self.store.append(event_type, payload)
        SessionReducer.apply(self.state, event)

    def accept(
        self, name: str, raw_arguments: str, *, call_id: str = "provider-1"
    ) -> AcceptedToolCall:
        self.append(
            "assistant_accepted",
            {
                "tool_calls": [
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {"name": name, "arguments": raw_arguments},
                    }
                ]
            },
        )
        return AcceptedToolCall(
            call_key=f"{self.state.last_seq}:{call_id}",
            provider_call_id=call_id,
            name=name,
            raw_arguments=raw_arguments,
        )

    def events_after_acceptance(self) -> list[object]:
        return self.store.load()[3:]

    def executor(
        self,
        *,
        registry: ToolRegistry | None = None,
        approver: object | None = None,
    ) -> ToolExecutor:
        return ToolExecutor(
            registry=registry or create_tool_registry(self.workspace),
            store=self.store,
            state=self.state,
            approver=approver or RecordingApprover(ApprovalDecision.ALLOW_ONCE),
            workspace=self.workspace,
        )


class ToolExecutorTests(ExecutorTestCase):
    def test_unknown_tool_finishes_without_approval_or_start(self) -> None:
        call = self.accept("missing", "{}")
        approver = Mock(side_effect=AssertionError("approval must not run"))

        result = self.executor(approver=approver).execute(call)

        events = self.events_after_acceptance()
        self.assertEqual([event.type for event in events], ["tool_finished"])
        self.assertEqual(events[0].payload["status"], "unknown_tool")
        self.assertEqual(events[0].payload["result"], "unknown tool: missing")
        self.assertEqual(result.status, "unknown_tool")
        approver.assert_not_called()
        self.assertEqual(
            self.state.tool_calls[call.call_key].status, ToolStatus.UNKNOWN_TOOL
        )

    def test_malformed_or_schema_invalid_arguments_finish_without_start(self) -> None:
        cases = (
            ("{", "arguments must be valid JSON"),
            ('{"path":3}', "path must be a string"),
        )
        for raw_arguments, message in cases:
            with self.subTest(raw_arguments=raw_arguments):
                self.tearDown()
                self.setUp()
                call = self.accept("read_file", raw_arguments)
                approver = Mock(side_effect=AssertionError("approval must not run"))

                result = self.executor(approver=approver).execute(call)

                events = self.events_after_acceptance()
                self.assertEqual([event.type for event in events], ["tool_finished"])
                self.assertEqual(events[0].payload["status"], "invalid_arguments")
                self.assertIn(message, events[0].payload["result"])
                self.assertEqual(result.status, "invalid_arguments")
                approver.assert_not_called()

    def test_read_starts_executes_and_finishes_without_approval(self) -> None:
        (self.workspace / "notes.txt").write_text("hello\n", encoding="utf-8")
        call = self.accept("read_file", '{"path":"notes.txt"}')
        approver = Mock(side_effect=AssertionError("approval must not run"))

        result = self.executor(approver=approver).execute(call)

        events = self.events_after_acceptance()
        self.assertEqual(
            [event.type for event in events], ["tool_started", "tool_finished"]
        )
        self.assertEqual(result.status, "succeeded")
        self.assertEqual(events[-1].payload["result"], "1 | hello")
        approver.assert_not_called()

    def test_denied_write_records_decision_then_terminal_without_effect(self) -> None:
        path = self.workspace / "notes.txt"
        path.write_text("old\n", encoding="utf-8")
        call = self.accept(
            "write_file", '{"path":"notes.txt","content":"new\\n"}'
        )
        approver = RecordingApprover(ApprovalDecision.DENY)

        result = self.executor(approver=approver).execute(call)

        events = self.events_after_acceptance()
        self.assertEqual(
            [event.type for event in events],
            ["approval_decided", "tool_finished"],
        )
        self.assertIs(events[0].payload["approved"], False)
        self.assertEqual(events[0].payload["scope"], "once")
        self.assertEqual(events[1].payload["status"], "denied")
        self.assertEqual(path.read_text(encoding="utf-8"), "old\n")
        self.assertEqual(result.status, "denied")
        self.assertEqual(len(approver.requests), 1)
        self.assertIn(str(path.resolve()), str(approver.requests[0]))

    def test_write_orders_approval_snapshot_start_effect_and_success(self) -> None:
        path = self.workspace / "notes.txt"
        before = b"old\n"
        path.write_bytes(before)
        path.chmod(0o640)
        call = self.accept(
            "write_file", '{"path":"notes.txt","content":"new\\n"}'
        )
        approver = RecordingApprover(ApprovalDecision.ALLOW_ONCE)
        observed_before_effect: list[str] = []
        original_execute = PreparedFileChange.execute

        def execute(prepared: PreparedFileChange):
            observed_before_effect.extend(event.type for event in self.store.load()[3:])
            return original_execute(prepared)

        with patch.object(PreparedFileChange, "execute", execute):
            result = self.executor(approver=approver).execute(call)

        events = self.events_after_acceptance()
        self.assertEqual(
            [event.type for event in events],
            [
                "approval_decided",
                "file_snapshot",
                "tool_started",
                "tool_finished",
            ],
        )
        self.assertEqual(
            observed_before_effect,
            ["approval_decided", "file_snapshot", "tool_started"],
        )
        snapshot = events[1].payload
        self.assertEqual(snapshot["path"], str(path.resolve()))
        self.assertEqual(snapshot["before_bytes"], base64.b64encode(before).decode())
        self.assertEqual(snapshot["before_mode"], 0o640)
        expected_hash = hashlib.sha256(b"new\n").hexdigest()
        self.assertEqual(events[-1].payload["after_hash"], expected_hash)
        self.assertEqual(events[-1].payload["path"], str(path.resolve()))
        self.assertEqual(path.read_bytes(), b"new\n")
        self.assertEqual(result.metadata["after_hash"], expected_hash)
        self.assertEqual(
            self.state.file_snapshots[(self.turn_id, str(path.resolve()))].after_hash,
            expected_hash,
        )

    def test_second_write_uses_first_baseline_and_updates_latest_after_hash(self) -> None:
        path = self.workspace / "notes.txt"
        path.write_bytes(b"original")
        first = self.accept(
            "write_file", '{"path":"notes.txt","content":"middle"}', call_id="one"
        )
        executor = self.executor()
        executor.execute(first)
        self.append(
            "assistant_accepted",
            {
                "tool_calls": [
                    {
                        "id": "two",
                        "type": "function",
                        "function": {
                            "name": "write_file",
                            "arguments": '{"path":"notes.txt","content":"latest"}',
                        },
                    }
                ]
            },
        )
        second = AcceptedToolCall(
            call_key=f"{self.state.last_seq}:two",
            provider_call_id="two",
            name="write_file",
            raw_arguments='{"path":"notes.txt","content":"latest"}',
        )

        executor.execute(second)

        snapshots = [event for event in self.store.load() if event.type == "file_snapshot"]
        self.assertEqual(len(snapshots), 1)
        state_snapshot = self.state.file_snapshots[(self.turn_id, str(path.resolve()))]
        self.assertEqual(state_snapshot.before_bytes, base64.b64encode(b"original").decode())
        self.assertEqual(
            state_snapshot.after_hash, hashlib.sha256(b"latest").hexdigest()
        )

    def test_semantic_prepare_error_is_invalid_before_approval(self) -> None:
        call = self.accept(
            "edit_file", '{"path":"missing.txt","old_text":"a","new_text":"b"}'
        )
        approver = Mock(side_effect=AssertionError("approval must not run"))

        result = self.executor(approver=approver).execute(call)

        self.assertEqual(result.status, "invalid_arguments")
        self.assertEqual(
            [event.type for event in self.events_after_acceptance()],
            ["tool_finished"],
        )
        approver.assert_not_called()

    def test_execution_exception_finishes_failed_without_retry(self) -> None:
        calls = 0

        def fail(_: dict[str, object]) -> ToolResult:
            nonlocal calls
            calls += 1
            raise RuntimeError("boom")

        registry = ToolRegistry(
            [
                ToolSpec(
                    "explode",
                    "Fail once.",
                    {
                        "type": "object",
                        "properties": {},
                        "required": [],
                        "additionalProperties": False,
                    },
                    handler=fail,
                    side_effect=SideEffect.NONE,
                )
            ]
        )
        call = self.accept("explode", "{}")

        result = self.executor(registry=registry).execute(call)

        self.assertEqual(calls, 1)
        self.assertEqual(result.status, "failed")
        self.assertEqual(
            [event.type for event in self.events_after_acceptance()],
            ["tool_started", "tool_finished"],
        )
        self.assertEqual(
            self.events_after_acceptance()[-1].payload["result"],
            "tool execution failed: RuntimeError: boom",
        )

    def test_shell_result_status_and_metadata_are_persisted(self) -> None:
        call = self.accept("bash", '{"command":"exit 4"}')

        result = self.executor().execute(call)

        events = self.events_after_acceptance()
        self.assertEqual(
            [event.type for event in events],
            ["approval_decided", "tool_started", "tool_finished"],
        )
        self.assertEqual(result.status, "failed")
        self.assertEqual(events[-1].payload["status"], "failed")
        self.assertEqual(events[-1].payload["exit_code"], 4)
        self.assertIs(events[-1].payload["truncated"], False)

    def test_identity_mismatch_fails_loudly_before_any_new_event(self) -> None:
        call = self.accept("read_file", '{"path":"missing"}')
        mismatched = AcceptedToolCall(
            call_key=call.call_key,
            provider_call_id=call.provider_call_id,
            name="grep",
            raw_arguments=call.raw_arguments,
        )

        with self.assertRaisesRegex(ToolExecutorError, "identity mismatch"):
            self.executor().execute(mismatched)

        self.assertEqual(self.events_after_acceptance(), [])

    def test_reducer_failure_after_append_is_raised_loudly(self) -> None:
        (self.workspace / "notes.txt").write_text("hello", encoding="utf-8")
        call = self.accept("read_file", '{"path":"notes.txt"}')

        with patch(
            "mca.executor.SessionReducer.apply", side_effect=RuntimeError("bad reducer")
        ):
            with self.assertRaisesRegex(ToolExecutorError, "durable event"):
                self.executor().execute(call)

        self.assertEqual(
            [event.type for event in self.events_after_acceptance()], ["tool_started"]
        )


if __name__ == "__main__":
    unittest.main()
