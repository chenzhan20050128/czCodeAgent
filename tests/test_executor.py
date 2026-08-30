"""Tool execution pipeline and durable event ordering tests."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import sys
import tempfile
import unittest
import uuid
from pathlib import Path
from unittest.mock import Mock, patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mca.approval import ApprovalDecision, ApprovalRequest
from mca.domain import SessionReducer, SessionState, ToolStatus
from mca.executor import AcceptedToolCall, ToolExecutor, ToolExecutorError
from mca.store import RolloutStore
from mca.tools import create_tool_registry
from mca.tools.filesystem import PreparedFileChange
from mca.tools.registry import SideEffect, ToolRegistry, ToolResult, ToolSpec
from mca.tools.shell import BoundedOutputChannel


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

    def test_plan_mode_blocks_side_effecting_tools_before_approval(self) -> None:
        self.append("plan_mode_set", {"active": True})
        path = self.workspace / "notes.txt"
        path.write_text("old\n", encoding="utf-8")
        call = self.accept(
            "write_file", '{"path":"notes.txt","content":"new\\n"}'
        )
        before = len(self.store.load())
        approver = Mock(side_effect=AssertionError("approval must not run"))

        result = self.executor(approver=approver).execute(call)

        new_events = self.store.load()[before:]
        self.assertEqual([event.type for event in new_events], ["tool_finished"])
        self.assertEqual(result.status, "denied")
        self.assertIn("plan mode", result.output.lower())
        self.assertEqual(path.read_text(encoding="utf-8"), "old\n")
        approver.assert_not_called()

    def test_plan_mode_blocks_bash_before_approval(self) -> None:
        self.append("plan_mode_set", {"active": True})
        call = self.accept("bash", '{"command":"echo hi"}')
        approver = Mock(side_effect=AssertionError("approval must not run"))

        result = self.executor(approver=approver).execute(call)

        self.assertEqual(result.status, "denied")
        self.assertIn("plan mode", result.output.lower())
        approver.assert_not_called()

    def test_plan_mode_allows_read_only_tools(self) -> None:
        self.append("plan_mode_set", {"active": True})
        (self.workspace / "notes.txt").write_text("hello\n", encoding="utf-8")
        call = self.accept("read_file", '{"path":"notes.txt"}')

        result = self.executor().execute(call)

        self.assertEqual(result.status, "succeeded")
        self.assertEqual(result.output, "1 | hello")

    def test_session_scoped_always_approves_later_calls_and_records_scope(self) -> None:
        path = self.workspace / "notes.txt"
        path.write_text("old", encoding="utf-8")
        first = self.accept(
            "write_file", '{"path":"notes.txt","content":"new"}', call_id="one"
        )

        class SessionApprover:
            calls = 0

            def decide(self, request):
                del request
                self.calls += 1
                return ApprovalDecision.ALLOW_SESSION

        approver = SessionApprover()
        executor = self.executor(approver=approver)

        self.assertEqual(executor.execute(first).status, "succeeded")
        second = self.accept(
            "bash", '{"command":"true"}', call_id="two"
        )
        self.assertEqual(executor.execute(second).status, "succeeded")

        approvals = [
            event for event in self.store.load() if event.type == "approval_decided"
        ]
        self.assertEqual([event.payload["scope"] for event in approvals], ["session", "session"])
        self.assertEqual(approver.calls, 1)

    def test_replayed_session_scope_always_approves_without_prompting(self) -> None:
        path = self.workspace / "notes.txt"
        path.write_text("old", encoding="utf-8")
        first = self.accept(
            "write_file", '{"path":"notes.txt","content":"new"}', call_id="one"
        )
        class SessionApprover:
            def decide(self, request):
                del request
                return ApprovalDecision.ALLOW_SESSION
        self.executor(approver=SessionApprover()).execute(first)
        self.store.close()
        with RolloutStore.open(self.root / "sessions", self.session_id) as reopened:
            state = SessionReducer.replay(reopened.load())
            self.assertTrue(state.session_approval_always)

    def test_exit_plan_mode_approval_records_plan_mode_off_and_succeeds(self) -> None:
        self.append("plan_mode_set", {"active": True})
        call = self.accept(
            "exit_plan_mode", '{"plan":"# Fix\\nStep one."}'
        )
        approver = RecordingApprover(ApprovalDecision.ALLOW_ONCE)

        result = self.executor(approver=approver).execute(call)

        self.assertEqual(result.status, "succeeded")
        self.assertFalse(self.state.plan_mode_active)
        types = [event.type for event in self.events_after_acceptance()]
        self.assertIn("plan_mode_set", types)
        self.assertEqual(types[-1], "tool_finished")
        plan_off = [
            event
            for event in self.events_after_acceptance()
            if event.type == "plan_mode_set"
        ][-1]
        self.assertIs(plan_off.payload["active"], False)
        self.assertIn("# Fix", str(approver.requests[0]))

    def test_exit_plan_mode_denial_keeps_plan_mode_on(self) -> None:
        self.append("plan_mode_set", {"active": True})
        call = self.accept(
            "exit_plan_mode", '{"plan":"# Fix\\nStep one."}'
        )
        approver = RecordingApprover(ApprovalDecision.DENY)

        result = self.executor(approver=approver).execute(call)

        self.assertEqual(result.status, "denied")
        self.assertTrue(self.state.plan_mode_active)
        self.assertNotIn(
            "plan_mode_set",
            [
                event.type
                for event in self.events_after_acceptance()
                if event.payload.get("active") is False
            ],
        )

    def test_exit_plan_mode_rejects_a_plan_without_a_heading(self) -> None:
        self.append("plan_mode_set", {"active": True})
        call = self.accept("exit_plan_mode", '{"plan":"just prose"}')
        approver = Mock(side_effect=AssertionError("approval must not run"))

        result = self.executor(approver=approver).execute(call)

        self.assertEqual(result.status, "invalid_arguments")
        self.assertTrue(self.state.plan_mode_active)
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

    def test_approval_keyboard_interrupt_terminalizes_cancelled_then_propagates(self) -> None:
        path = self.workspace / "notes.txt"
        path.write_text("old", encoding="utf-8")
        call = self.accept(
            "write_file", '{"path":"notes.txt","content":"new"}'
        )
        approver = Mock()
        approver.decide.side_effect = KeyboardInterrupt

        with self.assertRaises(KeyboardInterrupt):
            self.executor(approver=approver).execute(call)

        events = self.events_after_acceptance()
        self.assertEqual(
            [event.type for event in events],
            ["approval_decided", "tool_finished"],
        )
        self.assertIs(events[0].payload["approved"], False)
        self.assertEqual(events[1].payload["status"], "cancelled")
        self.assertEqual(path.read_text(encoding="utf-8"), "old")

    def test_custom_side_effect_uses_its_approval_renderer(self) -> None:
        class Prepared:
            def execute(self) -> ToolResult:
                return ToolResult(title="custom", output="ran")

        registry = ToolRegistry(
            [
                ToolSpec(
                    "custom",
                    "Custom side effect.",
                    {
                        "type": "object",
                        "properties": {},
                        "required": [],
                        "additionalProperties": False,
                    },
                    prepare_handler=lambda _: Prepared(),
                    side_effect=True,
                    approval_renderer=lambda _: "Custom target\x1b[31m",
                )
            ]
        )
        approver = RecordingApprover(ApprovalDecision.ALLOW_ONCE)
        call = self.accept("custom", "{}")

        result = self.executor(registry=registry, approver=approver).execute(call)

        self.assertEqual(result.status, "succeeded")
        self.assertEqual(approver.requests[0].render(), r"Custom target\x1b[31m")

    def test_renderer_keyboard_interrupt_cancels_once_and_propagates(self) -> None:
        executions = 0

        class Prepared:
            def execute(self) -> ToolResult:
                nonlocal executions
                executions += 1
                return ToolResult(title="custom", output="ran")

        registry = ToolRegistry(
            [
                ToolSpec(
                    "custom",
                    "Custom side effect.",
                    {"type": "object", "properties": {}, "additionalProperties": False},
                    prepare_handler=lambda _: Prepared(),
                    side_effect=True,
                    approval_renderer=lambda _: (_ for _ in ()).throw(KeyboardInterrupt()),
                )
            ]
        )
        call = self.accept("custom", "{}")

        with self.assertRaises(KeyboardInterrupt):
            self.executor(registry=registry).execute(call)

        events = self.events_after_acceptance()
        self.assertEqual([event.type for event in events], ["tool_finished"])
        self.assertEqual(events[0].payload["status"], "cancelled")
        self.assertEqual(executions, 0)

    def test_renderer_failure_or_non_string_finishes_failed_once(self) -> None:
        for renderer in (
            lambda _: (_ for _ in ()).throw(RuntimeError("render failed")),
            lambda _: 42,
        ):
            with self.subTest(renderer=renderer):
                self.tearDown()
                self.setUp()
                executions = []

                class Prepared:
                    def execute(self) -> ToolResult:
                        executions.append(True)
                        return ToolResult(title="custom", output="ran")

                registry = ToolRegistry(
                    [
                        ToolSpec(
                            "custom",
                            "Custom side effect.",
                            {"type": "object", "properties": {}, "additionalProperties": False},
                            prepare_handler=lambda _: Prepared(),
                            side_effect=True,
                            approval_renderer=renderer,
                        )
                    ]
                )
                call = self.accept("custom", "{}")

                result = self.executor(registry=registry).execute(call)

                events = self.events_after_acceptance()
                self.assertEqual([event.type for event in events], ["tool_finished"])
                self.assertEqual(events[0].payload["status"], "failed")
                self.assertEqual(result.status, "failed")
                self.assertEqual(executions, [])

    def test_shell_executor_forwards_bounded_output_channel(self) -> None:
        channel = BoundedOutputChannel(capacity=8)
        call = self.accept("bash", '{"command":"printf live"}')
        executor = ToolExecutor(
            create_tool_registry(self.workspace),
            self.store,
            self.state,
            RecordingApprover(ApprovalDecision.ALLOW_ONCE),
            self.workspace,
            output_channel=channel,
        )

        executor.execute(call)

        self.assertIn(("stdout", "live"), channel.drain())

    def test_executor_rejects_callable_output_channel(self) -> None:
        with self.assertRaisesRegex(TypeError, "BoundedOutputChannel"):
            ToolExecutor(
                create_tool_registry(self.workspace),
                self.store,
                self.state,
                RecordingApprover(ApprovalDecision.ALLOW_ONCE),
                self.workspace,
                output_channel=lambda stream, text: None,
            )
        with self.assertRaisesRegex(TypeError, "unexpected keyword argument"):
            ToolExecutor(
                create_tool_registry(self.workspace),
                self.store,
                self.state,
                RecordingApprover(ApprovalDecision.ALLOW_ONCE),
                self.workspace,
                on_output=lambda stream, text: None,
            )

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
                "file_mutation_planned",
                "file_snapshot",
                "tool_started",
                "tool_finished",
            ],
        )
        self.assertEqual(
            observed_before_effect,
            [
                "approval_decided",
                "file_mutation_planned",
                "file_snapshot",
                "tool_started",
            ],
        )
        plan = events[1].payload
        snapshot = events[2].payload
        self.assertEqual(plan["call_key"], call.call_key)
        self.assertEqual(plan["path"], str(path.resolve()))
        self.assertEqual(plan["expected_version"]["sha256"], hashlib.sha256(before).hexdigest())
        self.assertEqual(plan["proposed_hash"], hashlib.sha256(b"new\n").hexdigest())
        self.assertIn("+new", plan["diff"])
        self.assertEqual(snapshot["path"], str(path.resolve()))
        self.assertEqual(snapshot["before_bytes"], base64.b64encode(before).decode())
        self.assertEqual(snapshot["before_mode"], 0o640)
        expected_hash = hashlib.sha256(b"new\n").hexdigest()
        self.assertEqual(events[-1].payload["after_hash"], expected_hash)
        self.assertEqual(events[-1].payload["after_mode"], 0o640)
        self.assertEqual(events[-1].payload["path"], str(path.resolve()))
        self.assertEqual(path.read_bytes(), b"new\n")
        self.assertEqual(result.metadata["after_hash"], expected_hash)
        self.assertEqual(result.metadata["after_mode"], 0o640)
        self.assertEqual(
            self.state.file_snapshots[(self.turn_id, str(path.resolve()))].after_hash,
            expected_hash,
        )
        self.assertEqual(
            self.state.file_snapshots[(self.turn_id, str(path.resolve()))].after_mode,
            0o640,
        )
        self.assertEqual(
            self.state.file_snapshots[
                (self.turn_id, str(path.resolve()))
            ].after_version.sha256,
            expected_hash,
        )
        self.assertIn(call.call_key, self.state.file_mutation_plans)

    def test_post_commit_interrupt_persists_success_then_propagates_signal(self) -> None:
        path = self.workspace / "interrupted.txt"
        path.write_text("before", encoding="utf-8")
        path.chmod(0o640)
        call = self.accept(
            "write_file", '{"path":"interrupted.txt","content":"after"}'
        )
        real_replace = os.replace

        def replace_then_interrupt(source: object, target: object) -> None:
            real_replace(source, target)
            raise KeyboardInterrupt

        with patch(
            "mca.tools.filesystem.os.replace", side_effect=replace_then_interrupt
        ):
            with self.assertRaises(KeyboardInterrupt) as raised:
                self.executor().execute(call)

        self.assertEqual(type(raised.exception).__name__, "PostCommitInterrupted")
        events = self.events_after_acceptance()
        self.assertEqual(events[-1].type, "tool_finished")
        self.assertEqual(events[-1].payload["status"], "succeeded")
        self.assertEqual(events[-1].payload["path"], str(path.resolve()))
        self.assertEqual(
            events[-1].payload["after_hash"], hashlib.sha256(b"after").hexdigest()
        )
        self.assertEqual(events[-1].payload["after_mode"], 0o640)
        self.assertEqual(self.state.tool_calls[call.call_key].status, ToolStatus.SUCCEEDED)

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
        plans = [
            event
            for event in self.store.load()
            if event.type == "file_mutation_planned"
        ]
        self.assertEqual(len(snapshots), 1)
        self.assertEqual(len(plans), 2)
        self.assertEqual(
            {event.payload["call_key"] for event in plans},
            {first.call_key, second.call_key},
        )
        state_snapshot = self.state.file_snapshots[(self.turn_id, str(path.resolve()))]
        self.assertEqual(state_snapshot.before_bytes, base64.b64encode(b"original").decode())
        self.assertEqual(
            state_snapshot.after_hash, hashlib.sha256(b"latest").hexdigest()
        )
        self.assertEqual(
            state_snapshot.after_version.sha256,
            hashlib.sha256(b"latest").hexdigest(),
        )

    def test_successful_mutation_uses_its_own_plan_not_snapshot_source(self) -> None:
        path = self.workspace / "notes.txt"
        path.write_text("original", encoding="utf-8")
        self.append(
            "assistant_accepted",
            {
                "tool_calls": [
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": "write_file",
                            "arguments": '{"path":"notes.txt","content":"new"}',
                        },
                    }
                    for call_id in ("one", "two")
                ]
            },
        )
        first_key = f"{self.state.last_seq}:one"
        second_key = f"{self.state.last_seq}:two"
        prepared = create_tool_registry(self.workspace).resolve(
            "write_file"
        ).prepare_handler({"path": "notes.txt", "content": "new"})
        expected_version = prepared.expected_version.to_dict()
        for call_key, call_id in ((first_key, "one"), (second_key, "two")):
            self.append(
                "approval_decided",
                {
                    "call_key": call_key,
                    "call_id": call_id,
                    "approved": True,
                    "scope": "once",
                },
            )
            self.append(
                "file_mutation_planned",
                {
                    "turn_id": self.turn_id,
                    "call_key": call_key,
                    "path": str(path.resolve()),
                    "expected_version": expected_version,
                    "proposed_hash": hashlib.sha256(b"new").hexdigest(),
                    "diff": prepared.diff,
                },
            )
        self.append(
            "file_snapshot",
            {
                "turn_id": self.turn_id,
                "path": str(path.resolve()),
                "existed_before": True,
                "before_bytes": base64.b64encode(b"original").decode(),
                "before_encoding": "base64",
                "before_mode": 0o644,
                "call_key": first_key,
            },
        )
        self.append("tool_started", {"call_key": first_key})
        self.append("tool_started", {"call_key": second_key})
        after_version = {
            **expected_version,
            "sha256": hashlib.sha256(b"new").hexdigest(),
            "size": len(b"new"),
        }

        self.append(
            "tool_finished",
            {
                "call_key": second_key,
                "status": "succeeded",
                "result": "wrote",
                "path": str(path.resolve()),
                "after_hash": hashlib.sha256(b"new").hexdigest(),
                "after_mode": 0o644,
                "after_version": after_version,
            },
        )

        snapshot = self.state.file_snapshots[(self.turn_id, str(path.resolve()))]
        self.assertEqual(snapshot.source_call_key, first_key)
        self.assertEqual(snapshot.after_version.sha256, hashlib.sha256(b"new").hexdigest())

    def test_failed_first_write_does_not_freeze_a_stale_undo_baseline(self) -> None:
        path = self.workspace / "notes.txt"
        path.write_text("v1", encoding="utf-8")
        first = self.accept(
            "write_file", '{"path":"notes.txt","content":"first"}', call_id="one"
        )
        executor = self.executor()
        original_execute = PreparedFileChange.execute

        def conflict(prepared: PreparedFileChange):
            path.write_text("v2", encoding="utf-8")
            return original_execute(prepared)

        with patch.object(PreparedFileChange, "execute", conflict):
            result = executor.execute(first)
        self.assertEqual(result.status, "conflict")
        self.append(
            "assistant_accepted",
            {
                "tool_calls": [
                    {
                        "id": "two",
                        "type": "function",
                        "function": {
                            "name": "write_file",
                            "arguments": '{"path":"notes.txt","content":"v3"}',
                        },
                    }
                ]
            },
        )
        second = AcceptedToolCall(
            call_key=f"{self.state.last_seq}:two",
            provider_call_id="two",
            name="write_file",
            raw_arguments='{"path":"notes.txt","content":"v3"}',
        )

        executor.execute(second)

        snapshot = self.state.file_snapshots[(self.turn_id, str(path.resolve()))]
        self.assertEqual(snapshot.before_bytes, base64.b64encode(b"v2").decode())
        self.assertEqual(snapshot.after_hash, hashlib.sha256(b"v3").hexdigest())

    def test_post_replace_fsync_failure_is_recorded_as_committed_success(self) -> None:
        path = self.workspace / "committed.txt"
        path.write_text("before", encoding="utf-8")
        call = self.accept(
            "write_file", '{"path":"committed.txt","content":"after"}'
        )
        real_fsync = os.fsync
        fsync_calls = 0

        def fail_directory_fsync(descriptor: int) -> None:
            nonlocal fsync_calls
            fsync_calls += 1
            if fsync_calls == 2:
                raise OSError("directory fsync failed")
            real_fsync(descriptor)

        with patch(
            "mca.tools.filesystem._fsync_directory",
            side_effect=OSError("directory fsync failed"),
        ):
            result = self.executor().execute(call)

        self.assertEqual(path.read_text(encoding="utf-8"), "after")
        self.assertEqual(result.status, "succeeded")
        self.assertIs(result.metadata["durability_warning"], True)
        finished = self.events_after_acceptance()[-1]
        self.assertEqual(finished.payload["status"], "succeeded")
        self.assertEqual(
            finished.payload["after_hash"], hashlib.sha256(b"after").hexdigest()
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

    def test_malformed_tool_result_still_gets_one_terminal_failure(self) -> None:
        def malformed(_: dict[str, object]) -> object:
            return object.__new__(ToolResult)

        invalid = malformed({})
        object.__setattr__(invalid, "title", "x")
        object.__setattr__(invalid, "output", "x")
        object.__setattr__(invalid, "status", "succeeded")
        object.__setattr__(invalid, "metadata", None)
        registry = ToolRegistry(
            [
                ToolSpec(
                    "malformed",
                    "Return malformed result.",
                    {
                        "type": "object",
                        "properties": {},
                        "required": [],
                        "additionalProperties": False,
                    },
                    handler=lambda _: invalid,
                    side_effect=SideEffect.NONE,
                )
            ]
        )
        call = self.accept("malformed", "{}")

        result = self.executor(registry=registry).execute(call)

        self.assertEqual(result.status, "failed")
        self.assertEqual(
            [event.type for event in self.events_after_acceptance()],
            ["tool_started", "tool_finished"],
        )
        self.assertEqual(
            self.state.tool_calls[call.call_key].status, ToolStatus.FAILED
        )

    def test_executor_rejects_registry_bound_to_another_workspace(self) -> None:
        other = self.root / "other"
        other.mkdir()
        with self.assertRaisesRegex(ValueError, "registry workspace"):
            self.executor(registry=create_tool_registry(other))

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

    def test_read_tool_path_safety_error_is_failed_not_conflict(self) -> None:
        outside = self.root / "outside.txt"
        outside.write_text("secret", encoding="utf-8")
        call = self.accept("read_file", '{"path":"../outside.txt"}')

        result = self.executor().execute(call)

        events = self.events_after_acceptance()
        self.assertEqual(
            [event.type for event in events], ["tool_started", "tool_finished"]
        )
        self.assertEqual(result.status, "failed")
        self.assertEqual(events[-1].payload["status"], "failed")
        self.assertIn("tool execution failed", events[-1].payload["result"])

    def test_list_dir_accepts_a_workspace_internal_absolute_path(self) -> None:
        (self.workspace / "child.txt").write_text("x", encoding="utf-8")
        call = self.accept(
            "list_dir", f'{{"path": {json.dumps(str(self.workspace))}}}'
        )

        result = self.executor().execute(call)

        self.assertEqual(result.status, "succeeded")
        self.assertIn("child.txt", result.output)

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

        executor = self.executor()
        with patch(
            "mca.executor.SessionReducer.apply", side_effect=RuntimeError("bad reducer")
        ):
            with self.assertRaisesRegex(ToolExecutorError, "durable event"):
                executor.execute(call)

        self.assertEqual(
            [event.type for event in self.events_after_acceptance()], ["tool_started"]
        )
        with self.assertRaisesRegex(ToolExecutorError, "unusable"):
            executor.execute(call)


if __name__ == "__main__":
    unittest.main()
