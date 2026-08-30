"""End-to-end tests for run_code over the real MCA tool pipeline."""

from __future__ import annotations

import json
import sys
import tempfile
import threading
import time
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mca.approval import ApprovalDecision
from mca.code_mode import CodeModeRunner
from mca.code_scheduler import CodeDagScheduler
from mca.code_graph import CodeRunStatus
from mca.domain import Event, SessionReducer, SessionState, ToolStatus
from mca.executor import AcceptedToolCall, ToolExecutor
from mca.store import RolloutStore
from mca.tools import create_tool_registry


class AllowApprover:
    def __init__(self) -> None:
        self.requests: list[object] = []

    def decide(self, request: object) -> ApprovalDecision:
        self.requests.append(request)
        return ApprovalDecision.ALLOW_ONCE


class SelectiveApprover:
    def __init__(self, denied_fragment: str) -> None:
        self.denied_fragment = denied_fragment
        self.requests: list[object] = []

    def decide(self, request: object) -> ApprovalDecision:
        self.requests.append(request)
        return (
            ApprovalDecision.DENY
            if self.denied_fragment in str(getattr(request, "target", ""))
            else ApprovalDecision.ALLOW_ONCE
        )


class InterruptingApprover:
    def decide(self, request: object) -> ApprovalDecision:
        raise KeyboardInterrupt


class FailingRuntime:
    def run(self, code, **kwargs):
        raise OSError("worker could not start")


class CancelledGraphRuntime:
    def run(self, code, *, execute_graph, cancellation_event, **kwargs):
        from mca.code_runtime import CodeRuntimeResult

        cancellation_event.set()
        execute_graph({
            "targets": ["node-1"],
            "nodes": [
                {"node_id": "node-1", "ordinal": 1, "name": "read_file", "arguments": {"path": "missing"}, "dependencies": []}
            ],
        })
        return CodeRuntimeResult(value={"caught": True})


class CodeModeIntegrationTests(unittest.TestCase):
    def test_multiline_string_is_written_without_parser_indentation(self) -> None:
        expected = (
            '"""Generated module."""\n\n'
            'def calculate(value):\n'
            '    return value + 1\n'
        )
        source = (
            "content = '''" + expected + "'''\n"
            'write = tools.write_file({"path": "generated.py", "content": content})\n'
            "return await write"
        )

        result = CodeModeRunner(
            store=self.store, state=self.state, executor=self.executor
        ).run(self.outer, description="write multiline source", code=source)

        self.assertEqual(result.status, "succeeded", result.output)
        self.assertEqual(
            (self.workspace / "generated.py").read_text(encoding="utf-8"),
            expected,
        )

    def test_realistic_multifile_codegen_uses_multiline_starred_and_comprehension(self) -> None:
        module_one = (
            '"""First generated module."""\n\n'
            'def add(a, b):\n'
            '    return a + b\n'
        )
        module_two = (
            '"""Second generated module."""\n\n'
            'def multiply(a, b):\n'
            '    return a * b\n'
        )
        test_source = (
            'import unittest\n'
            'from one import add\n'
            'from two import multiply\n\n'
            'class GeneratedTests(unittest.TestCase):\n'
            '    def test_values(self):\n'
            '        self.assertEqual(add(2, 3), 5)\n'
            '        self.assertEqual(multiply(2, 3), 6)\n'
        )
        source = (
            "ONE = '''" + module_one + "'''\n"
            "TWO = '''" + module_two + "'''\n"
            "TEST = '''" + test_source + "'''\n"
            'one = tools.write_file({"path": "one.py", "content": ONE})\n'
            'two = tools.write_file({"path": "two.py", "content": TWO})\n'
            'test = tools.write_file({"path": "test_generated.py", "content": TEST})\n'
            'writes = [one, two, test]\n'
            'verify = tools.bash({"command": "python3 -m unittest -v"}, after=writes)\n'
            'results = await gather(*writes, verify)\n'
            'return {"statuses": [result["status"] for result in results], "exit_code": results[-1]["exit_code"]}'
        )

        result = CodeModeRunner(
            store=self.store, state=self.state, executor=self.executor
        ).run(self.outer, description="realistic codegen", code=source)

        self.assertEqual(result.status, "succeeded", result.output)
        payload = json.loads(result.output)
        self.assertEqual(payload["result"]["statuses"], ["succeeded"] * 4)
        self.assertEqual(payload["result"]["exit_code"], 0)
        self.assertEqual((self.workspace / "one.py").read_text(), module_one)
        self.assertEqual((self.workspace / "two.py").read_text(), module_two)
        self.assertEqual((self.workspace / "test_generated.py").read_text(), test_source)

    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.workspace = (self.root / "work").resolve()
        self.workspace.mkdir()
        self.session_id = str(uuid.uuid4())
        self.turn_id = str(uuid.uuid4())
        self.store = RolloutStore.create(self.root / "sessions", self.session_id)
        self.addCleanup(self.store.close)
        self.state = SessionState()
        self.approver = AllowApprover()
        self.append("session_created", {"cwd": str(self.workspace), "model": "m", "context_window": 4096})
        self.append("turn_started", {"turn_id": self.turn_id, "user_input": "code"})
        self.append("assistant_accepted", {"content": None, "tool_calls": [{"id": "outer", "type": "function", "function": {"name": "run_code", "arguments": "{}"}}]})
        self.outer = AcceptedToolCall("3:outer", "outer", "run_code", "{}")
        self.append("tool_started", {"call_key": self.outer.call_key, "call_id": self.outer.call_id})
        self.executor = ToolExecutor(
            registry=create_tool_registry(self.workspace),
            store=self.store,
            state=self.state,
            approver=self.approver,
            workspace=self.workspace,
        )

    def test_registered_run_code_executes_through_tool_executor(self) -> None:
        # Rebuild the accepted outer identity with valid run_code arguments.
        self.store.close()
        session_id = str(uuid.uuid4())
        self.store = RolloutStore.create(self.root / "registered", session_id)
        self.state = SessionState()
        self.append("session_created", {"cwd": str(self.workspace), "model": "m", "context_window": 4096})
        self.append("turn_started", {"turn_id": self.turn_id, "user_input": "code"})
        arguments = json.dumps({"description": "return one", "code": "return 1"})
        self.append("assistant_accepted", {"content": None, "tool_calls": [{"id": "outer", "type": "function", "function": {"name": "run_code", "arguments": arguments}}]})
        executor = ToolExecutor(
            registry=create_tool_registry(self.workspace),
            store=self.store, state=self.state, approver=self.approver, workspace=self.workspace,
        )

        result = executor.execute(AcceptedToolCall("3:outer", "outer", "run_code", arguments))

        self.assertEqual(result.status, "succeeded", result.output)
        self.assertEqual(json.loads(result.output)["result"], 1)
        self.assertIs(self.state.tool_calls["3:outer"].status, ToolStatus.SUCCEEDED)

    def test_event_observer_sees_each_code_state_only_after_reduction(self) -> None:
        observed: list[tuple[str, str]] = []

        def observe(event: Event, state: SessionState) -> None:
            if event.type == "code_run_started":
                observed.append((event.type, state.code_runs[event.payload["run_id"]].status.value))
            elif event.type in {"code_node_planned", "tool_started", "tool_finished"}:
                call_key = event.payload.get("node_id", event.payload.get("call_key"))
                if isinstance(call_key, str) and call_key in state.code_nodes:
                    observed.append((event.type, state.code_nodes[call_key].status.value))
            elif event.type == "code_run_finished":
                observed.append((event.type, state.code_runs[event.payload["run_id"]].status.value))

        executor = ToolExecutor(
            registry=create_tool_registry(self.workspace),
            store=self.store,
            state=self.state,
            approver=self.approver,
            workspace=self.workspace,
            event_observer=observe,
        )

        result = CodeModeRunner(
            store=self.store, state=self.state, executor=executor
        ).run(
            self.outer,
            description="observe graph",
            code='return await tools.list_dir({"path": "."})',
        )

        self.assertEqual(result.status, "succeeded")
        self.assertEqual(
            observed,
            [
                ("code_run_started", CodeRunStatus.ACTIVE.value),
                ("code_node_planned", "planned"),
                ("tool_started", "started"),
                ("tool_finished", "succeeded"),
                ("code_run_finished", CodeRunStatus.SUCCEEDED.value),
            ],
        )

    def test_nested_result_preserves_tool_metadata_for_program(self) -> None:
        result = CodeModeRunner(
            store=self.store, state=self.state, executor=self.executor
        ).run(
            self.outer,
            description="read metadata",
            code='return await tools.list_dir({"path": "."})',
        )

        payload = json.loads(result.output)
        self.assertEqual(payload["result"]["metadata"]["path"], str(self.workspace))

    def test_parent_rejects_control_and_recursive_nested_tools(self) -> None:
        for forbidden in ("run_code", "exit_plan_mode"):
            with self.subTest(forbidden=forbidden):
                result = CodeModeRunner(
                    store=self.store, state=self.state, executor=self.executor
                ).run(
                    self.outer,
                    description="forbidden",
                    code=f'return await tools.{forbidden}({{}})',
                )
                self.assertEqual(result.status, "failed")
                self.assertIn("AttributeError", result.output)

    def test_node_limit_is_returned_as_graph_rejected(self) -> None:
        result = CodeModeRunner(
            store=self.store, state=self.state, executor=self.executor, max_nodes=1
        ).run(
            self.outer,
            description="too many nodes",
            code="""
a = tools.read_file({"path": "a"})
b = tools.read_file({"path": "b"})
try:
    return await gather(a, b)
except GraphExecutionError as error:
    return {"code": error.code}
""",
        )
        payload = json.loads(result.output)
        self.assertEqual(payload["result"], {"code": "GRAPH_REJECTED"})

    def test_unsubmitted_lazy_nodes_still_count_toward_node_limit(self) -> None:
        result = CodeModeRunner(
            store=self.store, state=self.state, executor=self.executor, max_nodes=1
        ).run(
            self.outer,
            description="too many lazy nodes",
            code=(
                'first = tools.read_file({"path": "a"})\n'
                'second = tools.read_file({"path": "b"})\n'
                'return {"created": 2}'
            ),
        )

        payload = json.loads(result.output)
        self.assertEqual(result.status, "failed")
        self.assertEqual(payload["runtime_error"]["code"], "NODE_LIMIT")
        self.assertEqual(payload["execution_summary"]["planned"], 0)

    def test_parent_rejects_forged_control_node_without_partial_plan(self) -> None:
        run_id = str(uuid.uuid4())
        self.append("code_run_started", {"run_id": run_id, "turn_id": self.turn_id, "parent_call_key": self.outer.call_key, "description": "forged", "source_hash": "sha256:x"})
        scheduler = CodeDagScheduler(store=self.store, state=self.state, executor=self.executor, run_id=run_id)

        with self.assertRaisesRegex(ValueError, "not available in Code Mode"):
            scheduler.execute_graph({
                "targets": ["node-2"],
                "nodes": [
                    {"node_id": "node-1", "ordinal": 1, "name": "read_file", "arguments": {"path": "a"}, "dependencies": []},
                    {"node_id": "node-2", "ordinal": 2, "name": "exit_plan_mode", "arguments": {"plan": "# x"}, "dependencies": ["node-1"]},
                ],
            })

        self.assertEqual(self.state.code_runs[run_id].node_ids, ())

    def test_duplicate_ordinal_rejects_whole_graph_before_append(self) -> None:
        run_id = str(uuid.uuid4())
        self.append("code_run_started", {"run_id": run_id, "turn_id": self.turn_id, "parent_call_key": self.outer.call_key, "description": "duplicate", "source_hash": "sha256:x"})
        scheduler = CodeDagScheduler(store=self.store, state=self.state, executor=self.executor, run_id=run_id)

        with self.assertRaisesRegex(ValueError, "ordinal"):
            scheduler.execute_graph({
                "targets": ["a", "b"],
                "nodes": [
                    {"node_id": "a", "ordinal": 1, "name": "read_file", "arguments": {"path": "a"}, "dependencies": []},
                    {"node_id": "b", "ordinal": 1, "name": "read_file", "arguments": {"path": "b"}, "dependencies": []},
                ],
            })

        self.assertEqual(self.state.code_runs[run_id].node_ids, ())

    def test_unknown_target_rejects_whole_graph_before_append(self) -> None:
        run_id = str(uuid.uuid4())
        self.append("code_run_started", {"run_id": run_id, "turn_id": self.turn_id, "parent_call_key": self.outer.call_key, "description": "unknown target", "source_hash": "sha256:x"})
        scheduler = CodeDagScheduler(
            store=self.store, state=self.state, executor=self.executor, run_id=run_id
        )

        with self.assertRaisesRegex(ValueError, "UNKNOWN_NODE_REFERENCE"):
            scheduler.execute_graph({
                "targets": ["missing"],
                "nodes": [
                    {"node_id": "valid", "ordinal": 1, "name": "read_file", "arguments": {"path": "a"}, "dependencies": []},
                ],
            })

        self.assertEqual(self.state.code_runs[run_id].node_ids, ())

    def append(self, event_type: str, payload: dict[str, object]) -> None:
        event = self.store.append(event_type, payload)
        SessionReducer.apply(self.state, event)

    def test_parallel_writes_then_dependent_bash_complete(self) -> None:
        source = """
first = tools.write_file({"path": "a.txt", "content": "A"})
second = tools.write_file({"path": "b.txt", "content": "B"})
test = tools.bash({"command": "test $(cat a.txt)$(cat b.txt) = AB"}, after=[first, second])
return await test
"""

        result = CodeModeRunner(
            store=self.store, state=self.state, executor=self.executor
        ).run(self.outer, description="write and test", code=source)

        self.assertEqual(result.status, "succeeded", result.output)
        payload = json.loads(result.output)
        self.assertEqual(payload["execution_summary"]["succeeded"], 3)
        self.assertEqual((self.workspace / "a.txt").read_text(), "A")
        self.assertEqual((self.workspace / "b.txt").read_text(), "B")
        nodes = list(self.state.code_nodes.values())
        self.assertEqual([node.status.value for node in nodes], ["succeeded"] * 3)
        self.assertEqual(nodes[2].dependencies, (nodes[0].node_id, nodes[1].node_id))
        self.assertEqual(len(self.approver.requests), 3)

    def test_same_file_conflict_skips_dependent_but_unrelated_branch_completes(self) -> None:
        (self.workspace / "shared.txt").write_text("base", encoding="utf-8")
        source = """
first = tools.write_file({"path": "shared.txt", "content": "one"})
second = tools.write_file({"path": "shared.txt", "content": "two"})
blocked = tools.bash({"command": "touch should-not-exist"}, after=[first, second])
unrelated = tools.write_file({"path": "other.txt", "content": "ok"})
try:
    await execute(blocked, unrelated)
except GraphExecutionError as error:
    return {"caught": error.code}
"""

        result = CodeModeRunner(
            store=self.store, state=self.state, executor=self.executor
        ).run(self.outer, description="conflict graph", code=source)

        self.assertEqual(result.status, "failed")
        payload = json.loads(result.output)
        self.assertEqual(payload["execution_summary"]["conflict"], 1)
        self.assertEqual(payload["execution_summary"]["upstream_failed"], 1)
        self.assertEqual(payload["execution_summary"]["succeeded"], 2)
        self.assertFalse((self.workspace / "should-not-exist").exists())
        self.assertEqual((self.workspace / "other.txt").read_text(), "ok")
        statuses = {node.name + str(node.ordinal): node.status for node in self.state.code_nodes.values()}
        self.assertIn(ToolStatus.CONFLICT.value, {status.value for status in statuses.values()})
        self.assertIn(ToolStatus.UPSTREAM_FAILED.value, {status.value for status in statuses.values()})

    def test_same_path_parallel_writes_commit_in_node_ordinal_order(self) -> None:
        (self.workspace / "shared.txt").write_text("base", encoding="utf-8")
        original_execute = self.executor.dispatch_staged_with_cancel

        def delay_first(staged, cancellation_event=None):
            if staged.call.ordinal == 1:
                time.sleep(0.08)
            return original_execute(staged, cancellation_event)

        self.executor.dispatch_staged_with_cancel = delay_first
        result = CodeModeRunner(
            store=self.store, state=self.state, executor=self.executor
        ).run(
            self.outer,
            description="ordered conflict",
            code="""
first = tools.write_file({"path": "shared.txt", "content": "first"})
second = tools.write_file({"path": "shared.txt", "content": "second"})
try:
    return await gather(first, second)
except ToolCallError as error:
    return {"code": error.code}
""",
        )

        self.assertEqual(result.status, "failed")
        nodes = list(self.state.code_nodes.values())
        self.assertEqual(nodes[0].status.value, "succeeded")
        self.assertEqual(nodes[1].status.value, "conflict")
        self.assertEqual((self.workspace / "shared.txt").read_text(), "first")

    def test_cancellation_after_durable_start_never_leaves_a_started_node(self) -> None:
        run_id = str(uuid.uuid4())
        self.append("code_run_started", {"run_id": run_id, "turn_id": self.turn_id, "parent_call_key": self.outer.call_key, "description": "cancel window", "source_hash": "sha256:x"})
        cancellation = threading.Event()

        def cancel_after_start(event: Event, state: SessionState) -> None:
            if event.type == "tool_started" and event.payload.get("origin") == "code":
                cancellation.set()

        executor = ToolExecutor(
            registry=create_tool_registry(self.workspace),
            store=self.store, state=self.state, approver=self.approver,
            workspace=self.workspace, event_observer=cancel_after_start,
        )
        scheduler = CodeDagScheduler(
            store=self.store, state=self.state, executor=executor, run_id=run_id,
            max_parallel=2, cancellation_event=cancellation,
        )

        response = scheduler.execute_graph({
            "targets": ["a", "b"],
            "nodes": [
                {"node_id": "a", "ordinal": 1, "name": "list_dir", "arguments": {"path": "."}, "dependencies": []},
                {"node_id": "b", "ordinal": 2, "name": "list_dir", "arguments": {"path": "."}, "dependencies": []},
            ],
        })

        self.assertTrue(self.state.code_nodes[f"{run_id}:node:1"].status.is_terminal)
        self.assertEqual(self.state.code_nodes[f"{run_id}:node:2"].status.value, "not_executed")
        self.assertIn("a", response["results"])
        self.assertIn("b", response["results"])

    def test_cancellation_does_not_start_a_deferred_same_path_write(self) -> None:
        (self.workspace / "shared.txt").write_text("base", encoding="utf-8")
        run_id = str(uuid.uuid4())
        self.append("code_run_started", {"run_id": run_id, "turn_id": self.turn_id, "parent_call_key": self.outer.call_key, "description": "cancel deferred", "source_hash": "sha256:x"})
        cancellation = threading.Event()

        def cancel_after_first_start(event: Event, state: SessionState) -> None:
            if event.type == "tool_started" and event.payload.get("origin") == "code":
                cancellation.set()

        executor = ToolExecutor(
            registry=create_tool_registry(self.workspace),
            store=self.store, state=self.state, approver=self.approver,
            workspace=self.workspace, event_observer=cancel_after_first_start,
        )
        scheduler = CodeDagScheduler(
            store=self.store, state=self.state, executor=executor, run_id=run_id,
            max_parallel=2, cancellation_event=cancellation,
        )

        scheduler.execute_graph({
            "targets": ["a", "b"],
            "nodes": [
                {"node_id": "a", "ordinal": 1, "name": "write_file", "arguments": {"path": "shared.txt", "content": "one"}, "dependencies": []},
                {"node_id": "b", "ordinal": 2, "name": "write_file", "arguments": {"path": "shared.txt", "content": "two"}, "dependencies": []},
            ],
        })

        first = self.state.code_nodes[f"{run_id}:node:1"]
        second = self.state.code_nodes[f"{run_id}:node:2"]
        self.assertEqual(first.status.value, "cancelled")
        self.assertEqual(second.status.value, "not_executed")
        self.assertIsNone(second.started_seq)

    def test_keyboard_interrupt_while_waiting_cancels_inflight_bash_promptly(self) -> None:
        from concurrent.futures import wait as real_wait

        source = """
slow = tools.bash({"command": "sleep 5; touch too-late", "timeout_seconds": 10})
return await slow
"""
        started = time.monotonic()
        calls = 0

        def interrupt_once(futures, *args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise KeyboardInterrupt
            return real_wait(futures, *args, **kwargs)

        with patch("mca.code_scheduler.wait", side_effect=interrupt_once):
            with self.assertRaises(KeyboardInterrupt):
                CodeModeRunner(
                    store=self.store, state=self.state, executor=self.executor
                ).run(self.outer, description="interrupt wait", code=source)

        self.assertLess(time.monotonic() - started, 1.5)
        self.assertFalse((self.workspace / "too-late").exists())
        node = next(iter(self.state.code_nodes.values()))
        self.assertEqual(node.status.value, "interrupted")
        self.assertEqual(next(iter(self.state.code_runs.values())).status.value, "interrupted")

    def test_plan_mode_denies_nested_write_and_skips_its_dependent(self) -> None:
        self.append("plan_mode_set", {"active": True})
        source = """
write = tools.write_file({"path": "blocked.txt", "content": "x"})
test = tools.bash({"command": "touch should-not-exist"}, after=[write])
try:
    await test
except GraphExecutionError as error:
    return {"code": error.code}
"""

        result = CodeModeRunner(
            store=self.store, state=self.state, executor=self.executor
        ).run(self.outer, description="plan blocked", code=source)

        payload = json.loads(result.output)
        self.assertEqual(payload["execution_summary"]["denied"], 1)
        self.assertEqual(payload["execution_summary"]["upstream_failed"], 1)
        self.assertFalse((self.workspace / "blocked.txt").exists())
        self.assertFalse((self.workspace / "should-not-exist").exists())
        self.assertEqual(self.approver.requests, [])

    def test_denied_node_blocks_descendant_but_unrelated_write_succeeds(self) -> None:
        approver = SelectiveApprover("blocked.txt")
        self.executor = ToolExecutor(
            registry=create_tool_registry(self.workspace),
            store=self.store, state=self.state, approver=approver, workspace=self.workspace,
        )
        source = """
blocked = tools.write_file({"path": "blocked.txt", "content": "x"})
dependent = tools.bash({"command": "touch should-not-exist"}, after=[blocked])
unrelated = tools.write_file({"path": "ok.txt", "content": "ok"})
try:
    await execute(dependent, unrelated)
except GraphExecutionError:
    return {"handled": True}
"""

        result = CodeModeRunner(
            store=self.store, state=self.state, executor=self.executor
        ).run(self.outer, description="deny branch", code=source)

        payload = json.loads(result.output)
        self.assertEqual(payload["execution_summary"]["denied"], 1)
        self.assertEqual(payload["execution_summary"]["upstream_failed"], 1)
        self.assertEqual(payload["execution_summary"]["succeeded"], 1)
        self.assertFalse((self.workspace / "blocked.txt").exists())
        self.assertEqual((self.workspace / "ok.txt").read_text(), "ok")

    def test_two_bash_nodes_declared_parallel_overlap(self) -> None:
        source = """
first = tools.bash({"command": "sleep 0.2; echo one"})
second = tools.bash({"command": "sleep 0.2; echo two"})
return await gather(first, second)
"""

        started = time.monotonic()
        result = CodeModeRunner(
            store=self.store, state=self.state, executor=self.executor
        ).run(self.outer, description="parallel shell", code=source)
        elapsed = time.monotonic() - started

        self.assertEqual(result.status, "succeeded", result.output)
        self.assertLess(elapsed, 0.38)
        self.assertEqual(len(self.approver.requests), 2)
        self.assertEqual(
            [node.status.value for node in self.state.code_nodes.values()],
            ["succeeded", "succeeded"],
        )

    def test_code_wall_timeout_interrupts_inflight_bash_and_skips_dependent(self) -> None:
        source = """
slow = tools.bash({"command": "sleep 5; touch too-late", "timeout_seconds": 10})
after = tools.write_file({"path": "after.txt", "content": "bad"}, after=[slow])
return await after
"""
        from mca.code_runtime import CodeRuntime, CodeRuntimeConfig

        started = time.monotonic()
        result = CodeModeRunner(
            store=self.store,
            state=self.state,
            executor=self.executor,
            runtime=CodeRuntime(CodeRuntimeConfig(max_wall_seconds=0.1)),
        ).run(self.outer, description="timeout graph", code=source)
        elapsed = time.monotonic() - started

        self.assertLess(elapsed, 1.5)
        self.assertEqual(result.status, "failed")
        self.assertFalse((self.workspace / "too-late").exists())
        self.assertFalse((self.workspace / "after.txt").exists())
        statuses = [node.status.value for node in self.state.code_nodes.values()]
        self.assertEqual(statuses, ["interrupted", "upstream_failed"])

    def test_nested_approval_interrupt_closes_run_and_unstarted_sibling(self) -> None:
        executor = ToolExecutor(
            registry=create_tool_registry(self.workspace),
            store=self.store, state=self.state, approver=InterruptingApprover(),
            workspace=self.workspace,
        )
        source = """
first = tools.write_file({"path": "a.txt", "content": "A"})
second = tools.write_file({"path": "b.txt", "content": "B"})
return await gather(first, second)
"""

        with self.assertRaises(KeyboardInterrupt):
            CodeModeRunner(
                store=self.store, state=self.state, executor=executor
            ).run(self.outer, description="interrupt approval", code=source)

        run = next(reversed(self.state.code_runs.values()))
        self.assertEqual(run.status.value, "interrupted")
        self.assertEqual(
            [self.state.code_nodes[node_id].status.value for node_id in run.node_ids],
            ["cancelled", "not_executed"],
        )

    def test_oversized_graph_response_fails_and_closes_code_run(self) -> None:
        from mca.code_runtime import CodeRuntime, CodeRuntimeConfig

        (self.workspace / "large.txt").write_text("x" * 4096, encoding="utf-8")
        result = CodeModeRunner(
            store=self.store,
            state=self.state,
            executor=self.executor,
            runtime=CodeRuntime(
                CodeRuntimeConfig(
                    max_wall_seconds=2,
                    max_output_bytes=8192,
                    max_frame_bytes=1024,
                )
            ),
        ).run(
            self.outer,
            description="large graph result",
            code='return await tools.read_file({"path": "large.txt"})',
        )

        self.assertEqual(result.status, "failed")
        payload = json.loads(result.output)
        self.assertEqual(payload["runtime_error"]["code"], "PROTOCOL_ERROR")
        run = next(reversed(self.state.code_runs.values()))
        self.assertEqual(run.status.value, "failed")
        self.assertTrue(all(node.status.is_terminal for node in self.state.code_nodes.values()))

    def test_runtime_exception_closes_code_run_before_propagating(self) -> None:
        runner = CodeModeRunner(
            store=self.store, state=self.state, executor=self.executor,
            runtime=FailingRuntime(),
        )

        with self.assertRaisesRegex(OSError, "worker could not start"):
            runner.run(self.outer, description="startup failure", code="return 1")

        run = next(reversed(self.state.code_runs.values()))
        self.assertEqual(run.status.value, "failed")

    def test_post_append_interrupt_synchronizes_before_closing_code_run(self) -> None:
        original_append = self.store.append
        interrupted = False

        def append_then_interrupt(event_or_type, payload=None):
            nonlocal interrupted
            event = original_append(event_or_type, payload)
            event_type = event.type
            if (
                not interrupted
                and event_type == "tool_finished"
                and event.payload.get("origin") == "code"
                and event.payload.get("status") == "succeeded"
            ):
                interrupted = True
                raise KeyboardInterrupt
            return event

        with patch.object(self.store, "append", side_effect=append_then_interrupt):
            with self.assertRaises(KeyboardInterrupt):
                CodeModeRunner(
                    store=self.store, state=self.state, executor=self.executor
                ).run(
                    self.outer, description="post append interrupt",
                    code='return await tools.list_dir({"path": "."})',
                )

        replayed = SessionReducer.replay(self.store.load())
        self.assertEqual(replayed, self.state)
        node = next(iter(self.state.code_nodes.values()))
        self.assertEqual(node.status.value, "succeeded")
        run = next(iter(self.state.code_runs.values()))
        self.assertEqual(run.status.value, "interrupted")

    def test_caught_non_success_node_still_forces_outer_failure(self) -> None:
        result = CodeModeRunner(
            store=self.store, state=self.state, executor=self.executor,
            runtime=CancelledGraphRuntime(),
        ).run(self.outer, description="caught cancellation", code="return 1")

        self.assertEqual(result.status, "failed")
        payload = json.loads(result.output)
        self.assertEqual(payload["execution_summary"]["not_executed"], 1)
        self.assertEqual(next(iter(self.state.code_runs.values())).status.value, "failed")

    def test_run_code_composes_all_six_ordinary_tools(self) -> None:
        (self.workspace / "source.txt").write_text("needle\n", encoding="utf-8")
        (self.workspace / "edit.txt").write_text("old\n", encoding="utf-8")
        source = """
listed = tools.list_dir({"path": "."})
read = tools.read_file({"path": "source.txt"})
searched = tools.grep({"pattern": "needle", "path": "source.txt"})
written = tools.write_file({"path": "new.txt", "content": "new"})
edited = tools.edit_file({"path": "edit.txt", "old_text": "old", "new_text": "changed"})
verified = tools.bash({"command": "test $(cat new.txt) = new && test $(cat edit.txt) = changed"}, after=[listed, read, searched, written, edited])
return await verified
"""

        result = CodeModeRunner(
            store=self.store, state=self.state, executor=self.executor
        ).run(self.outer, description="all tools", code=source)

        self.assertEqual(result.status, "succeeded", result.output)
        payload = json.loads(result.output)
        self.assertEqual(payload["execution_summary"]["succeeded"], 6)
        self.assertEqual(len(self.approver.requests), 3)
        names = [node.name for node in self.state.code_nodes.values()]
        self.assertEqual(
            names,
            ["list_dir", "read_file", "grep", "write_file", "edit_file", "bash"],
        )
        bash_node = list(self.state.code_nodes.values())[-1]
        self.assertEqual(len(bash_node.dependencies), 5)


if __name__ == "__main__":
    unittest.main()
