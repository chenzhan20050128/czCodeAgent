"""End-to-end tests for run_code over the real MCA tool pipeline."""

from __future__ import annotations

import json
import sys
import tempfile
import time
import unittest
import uuid
from pathlib import Path

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


class CodeModeIntegrationTests(unittest.TestCase):
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
