"""Dynamic DAG scheduling for one active Code Mode run."""

from __future__ import annotations

import json
import threading
from collections.abc import Mapping
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from typing import Any

from .code_graph import CodeNodeStatus
from .domain import Event, SessionReducer, SessionState, ToolCall, ToolStatus
from .executor import AcceptedToolCall, PreparedStagedCall, ToolExecutor
from .store import RolloutStore
from .tools.registry import ToolResult


_SUCCESS = {ToolStatus.SUCCEEDED, ToolStatus.USER_CONFIRMED_SUCCESS}
_CODE_TOOLS = frozenset(
    {"read_file", "list_dir", "grep", "write_file", "edit_file", "bash"}
)


@dataclass(frozen=True)
class _PreparedNode:
    local_id: str
    durable_id: str
    staged: PreparedStagedCall


class CodeDagScheduler:
    """Plan and execute graph closures through the shared tool pipeline."""

    def __init__(
        self,
        *,
        store: RolloutStore,
        state: SessionState,
        executor: ToolExecutor,
        run_id: str,
        max_parallel: int = 4,
        max_nodes: int = 64,
        cancellation_event: threading.Event | None = None,
    ) -> None:
        if type(max_parallel) is not int or max_parallel < 1:
            raise ValueError("max_parallel must be positive")
        if type(max_nodes) is not int or max_nodes < 1:
            raise ValueError("max_nodes must be positive")
        self.store = store
        self.state = state
        self.executor = executor
        self.run_id = run_id
        self.max_parallel = max_parallel
        self.max_nodes = max_nodes
        self.cancellation_event = cancellation_event or threading.Event()
        self.local_to_durable: dict[str, str] = {}

    def execute_graph(self, request: dict[str, object]) -> dict[str, object]:
        nodes = request.get("nodes")
        targets = request.get("targets")
        if not isinstance(nodes, list) or not isinstance(targets, list):
            raise ValueError("execute_graph requires nodes and targets arrays")
        if len(self.local_to_durable) + len(nodes) > self.max_nodes:
            raise ValueError("code graph exceeds node limit")
        normalized = self._normalize_nodes(nodes)
        ordered = self._topological(normalized)
        self._validate_batch(ordered)
        for node in ordered:
            self._plan(node)
        target_ids = [self._durable_id(value) for value in targets]
        relevant = self._closure(target_ids)
        self._execute_closure(relevant)
        return {
            "results": {
                local_id: self._outcome(
                    self.state.tool_calls[durable_id],
                    self.state.code_nodes[durable_id],
                )
                for local_id, durable_id in self.local_to_durable.items()
                if durable_id in relevant
            }
        }

    def _normalize_nodes(
        self, raw_nodes: list[object]
    ) -> dict[str, dict[str, Any]]:
        normalized: dict[str, dict[str, Any]] = {}
        for raw in raw_nodes:
            if not isinstance(raw, dict):
                raise ValueError("code graph node must be an object")
            local_id = raw.get("node_id")
            ordinal = raw.get("ordinal")
            name = raw.get("name")
            arguments = raw.get("arguments")
            dependencies = raw.get("dependencies", [])
            if (
                not isinstance(local_id, str)
                or not local_id
                or type(ordinal) is not int
                or ordinal < 1
                or not isinstance(name, str)
                or not name
                or not isinstance(arguments, dict)
                or not isinstance(dependencies, list)
                or any(not isinstance(item, str) for item in dependencies)
            ):
                raise ValueError("invalid code graph node")
            if local_id in self.local_to_durable or local_id in normalized:
                raise ValueError("code graph node id already submitted")
            normalized[local_id] = {
                "local_id": local_id,
                "ordinal": ordinal,
                "name": name,
                "arguments": arguments,
                "dependencies": tuple(dependencies),
            }
        return normalized

    def _validate_batch(self, nodes: list[dict[str, Any]]) -> None:
        ordinals = [node["ordinal"] for node in nodes]
        if len(ordinals) != len(set(ordinals)):
            raise ValueError("code graph node ordinal must be unique")
        durable_ids = [f"{self.run_id}:node:{ordinal}" for ordinal in ordinals]
        if len(durable_ids) != len(set(durable_ids)) or any(
            durable_id in self.state.code_nodes or durable_id in self.state.tool_calls
            for durable_id in durable_ids
        ):
            raise ValueError("code graph durable node id already exists")
        for node in nodes:
            if node["name"] not in _CODE_TOOLS:
                raise ValueError(
                    f"tool {node['name']!r} is not available in Code Mode"
                )

    def _topological(
        self, nodes: dict[str, dict[str, Any]]
    ) -> list[dict[str, Any]]:
        temporary: set[str] = set()
        permanent: set[str] = set()
        ordered: list[dict[str, Any]] = []

        def visit(local_id: str) -> None:
            if local_id in permanent or local_id in self.local_to_durable:
                return
            if local_id in temporary:
                raise ValueError("CYCLIC_DEPENDENCY")
            node = nodes.get(local_id)
            if node is None:
                raise ValueError("UNKNOWN_NODE_REFERENCE")
            temporary.add(local_id)
            for dependency in node["dependencies"]:
                visit(dependency)
            temporary.remove(local_id)
            permanent.add(local_id)
            ordered.append(node)

        for node in sorted(nodes.values(), key=lambda value: value["ordinal"]):
            visit(node["local_id"])
        return ordered

    def _plan(self, node: dict[str, Any]) -> None:
        run = self.state.code_runs[self.run_id]
        local_id = node["local_id"]
        durable_id = f"{self.run_id}:node:{node['ordinal']}"
        dependencies = [self._durable_id(item) for item in node["dependencies"]]
        encoded = json.dumps(
            node["arguments"],
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        event = self.store.append(
            "code_node_planned",
            {
                "run_id": self.run_id,
                "node_id": durable_id,
                "ordinal": node["ordinal"],
                "name": node["name"],
                "arguments": encoded,
                "dependencies": dependencies,
            },
        )
        SessionReducer.apply(self.state, event)
        self.executor.observe_event(event)
        self.local_to_durable[local_id] = durable_id

    def _durable_id(self, local_id: object) -> str:
        if not isinstance(local_id, str) or local_id not in self.local_to_durable:
            raise ValueError("UNKNOWN_NODE_REFERENCE")
        return self.local_to_durable[local_id]

    def _closure(self, targets: list[str]) -> set[str]:
        included: set[str] = set()

        def visit(node_id: str) -> None:
            if node_id in included:
                return
            node = self.state.code_nodes[node_id]
            included.add(node_id)
            for dependency in node.dependencies:
                visit(dependency)

        for target in targets:
            visit(target)
        return included

    def _execute_closure(self, relevant: set[str]) -> None:
        while True:
            pending = [
                self.state.code_nodes[node_id]
                for node_id in relevant
                if self.state.code_nodes[node_id].status is CodeNodeStatus.PLANNED
            ]
            if not pending:
                return
            progress = False
            for node in sorted(pending, key=lambda value: value.ordinal):
                blockers = [
                    dependency
                    for dependency in node.dependencies
                    if self.state.tool_calls[dependency].status not in _SUCCESS
                    and self.state.tool_calls[dependency].is_terminal
                ]
                if blockers:
                    roots: list[str] = []
                    for blocker in blockers:
                        source = self.state.code_nodes[blocker]
                        for root in source.root_failures or (blocker,):
                            if root not in roots:
                                roots.append(root)
                    self.executor.finish_unstarted(
                        self.state.tool_calls[node.node_id],
                        ToolStatus.UPSTREAM_FAILED,
                        "node was not executed because a dependency failed",
                        extra={"blocked_by": blockers, "root_failures": roots},
                    )
                    progress = True
            if self.cancellation_event.is_set():
                remaining = [
                    self.state.code_nodes[node_id]
                    for node_id in relevant
                    if self.state.code_nodes[node_id].status is CodeNodeStatus.PLANNED
                ]
                for node in sorted(remaining, key=lambda value: value.ordinal):
                    self.executor.finish_unstarted(
                        self.state.tool_calls[node.node_id],
                        ToolStatus.NOT_EXECUTED,
                        "node was not started because the code run was cancelled",
                    )
                return
            ready = [
                self.state.code_nodes[node_id]
                for node_id in relevant
                if self.state.code_nodes[node_id].status is CodeNodeStatus.PLANNED
                and all(
                    self.state.tool_calls[dependency].status in _SUCCESS
                    for dependency in self.state.code_nodes[node_id].dependencies
                )
            ]
            if ready:
                self._execute_ready(sorted(ready, key=lambda value: value.ordinal))
                progress = True
            if not progress:
                raise RuntimeError("code graph made no progress")

    def _execute_ready(self, nodes: list[Any]) -> None:
        staged: list[_PreparedNode] = []
        for node in nodes:
            if self.cancellation_event.is_set():
                self.executor.finish_unstarted(
                    self.state.tool_calls[node.node_id],
                    ToolStatus.NOT_EXECUTED,
                    "node was not started because the code run was cancelled",
                )
                continue
            call = self.state.tool_calls[node.node_id]
            prepared = self.executor.prepare_staged(
                AcceptedToolCall.from_tool_call(call),
                allow_code_concurrency=True,
            )
            if isinstance(prepared, ToolResult):
                continue
            staged.append(_PreparedNode(self._local_id(node.node_id), node.node_id, prepared))
        if not staged:
            return
        with ThreadPoolExecutor(
            max_workers=min(self.max_parallel, len(staged)),
            thread_name_prefix="mca-code-node",
        ) as pool:
            in_flight: dict[Future[ToolResult], _PreparedNode] = {}
            next_to_start = 0
            while next_to_start < len(staged) or in_flight:
                while (
                    next_to_start < len(staged)
                    and len(in_flight) < self.max_parallel
                ):
                    item = staged[next_to_start]
                    self.executor.start_staged(item.staged)
                    if self.cancellation_event.is_set():
                        break
                    future = pool.submit(
                        self.executor.dispatch_staged_with_cancel,
                        item.staged,
                        self.cancellation_event,
                    )
                    in_flight[future] = item
                    next_to_start += 1
                if not in_flight:
                    break
                done, _ = wait(
                    tuple(in_flight),
                    timeout=0.05,
                    return_when=FIRST_COMPLETED,
                )
                if not done:
                    continue
                for future in done:
                    item = in_flight.pop(future)
                    self.executor.commit_staged(item.staged, future.result())
            for item in staged[next_to_start:]:
                self.executor.finish_unstarted(
                    self.state.tool_calls[item.durable_id],
                    ToolStatus.NOT_EXECUTED,
                    "node was not started because the code run was cancelled",
                )

    def _local_id(self, durable_id: str) -> str:
        for local_id, candidate in self.local_to_durable.items():
            if candidate == durable_id:
                return local_id
        raise KeyError(durable_id)

    @staticmethod
    def _outcome(call: ToolCall, node: Any) -> dict[str, object]:
        if call.status in _SUCCESS:
            return {
                "ok": True,
                "value": {
                    "status": call.status.value,
                    "output": call.result or "",
                    "exit_code": call.exit_code,
                    "truncated": call.truncated,
                    "metadata": _plain_json(call.result_metadata),
                },
            }
        node_error = {
            "status": call.status.value,
            "code": (
                "FILE_STALE_VERSION"
                if call.status is ToolStatus.CONFLICT
                else call.status.value.upper()
            ),
            "message": call.result or "tool call failed",
            **({"blocked_by": list(node.blocked_by)} if node.blocked_by else {}),
            **({"root_failures": list(node.root_failures)} if node.root_failures else {}),
        }
        return {"ok": False, "error": node_error}


def _plain_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _plain_json(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain_json(item) for item in value]
    return value


__all__ = ["CodeDagScheduler"]
