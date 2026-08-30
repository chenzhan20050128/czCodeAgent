"""Pure values and summaries for durable Code Mode execution graphs."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class CodeRunStatus(str, Enum):
    ACTIVE = "active"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    INTERRUPTED = "interrupted"


class CodeNodeStatus(str, Enum):
    PLANNED = "planned"
    STARTED = "started"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    DENIED = "denied"
    INVALID_ARGUMENTS = "invalid_arguments"
    UNKNOWN_TOOL = "unknown_tool"
    CONFLICT = "conflict"
    TIMED_OUT = "timed_out"
    INTERRUPTED = "interrupted"
    CANCELLED = "cancelled"
    NOT_EXECUTED = "not_executed"
    UPSTREAM_FAILED = "upstream_failed"
    OUTCOME_UNKNOWN = "outcome_unknown"
    ABANDONED = "abandoned"
    BATCH_LIMIT_EXCEEDED = "batch_limit_exceeded"
    USER_CONFIRMED_SUCCESS = "user_confirmed_success"
    USER_CONFIRMED_FAILURE = "user_confirmed_failure"

    @property
    def is_terminal(self) -> bool:
        return self not in {CodeNodeStatus.PLANNED, CodeNodeStatus.STARTED}


@dataclass(frozen=True)
class CodeRun:
    run_id: str
    turn_id: str
    parent_call_key: str
    description: str
    source_hash: str
    status: CodeRunStatus = CodeRunStatus.ACTIVE
    node_ids: tuple[str, ...] = ()
    result: str | None = None


@dataclass(frozen=True)
class CodeNode:
    node_id: str
    run_id: str
    parent_call_key: str
    turn_id: str
    ordinal: int
    name: str
    arguments: str
    dependencies: tuple[str, ...]
    status: CodeNodeStatus = CodeNodeStatus.PLANNED
    result: str | None = None
    blocked_by: tuple[str, ...] = ()
    root_failures: tuple[str, ...] = ()
    requested_seq: int | None = None
    started_seq: int | None = None
    finished_seq: int | None = None


@dataclass(frozen=True)
class CodeGraphNodeView:
    """Presentation-neutral view of one durable Code Mode node."""

    node_id: str
    ordinal: int
    name: str
    target: str
    dependency_ordinals: tuple[int, ...]
    dependent_ordinals: tuple[int, ...]
    status: str
    is_current: bool
    elapsed_ms: int | None
    result: str | None
    blocked_by_ordinals: tuple[int, ...]
    root_failure_ordinals: tuple[int, ...]
    approval: str | None = None


@dataclass(frozen=True)
class CodeGraphView:
    """A complete graph snapshot derived only from durable session state."""

    run_id: str
    description: str
    status: str
    nodes: tuple[CodeGraphNodeView, ...]
    summary: dict[str, Any]
    elapsed_ms: int | None
    shell_mutation_warning: bool


def graph_summary(state: Any, run_id: str) -> dict[str, Any]:
    """Return a stable JSON summary for one derived run."""

    run = state.code_runs[run_id]
    nodes = [state.code_nodes[node_id] for node_id in run.node_ids]
    summary: dict[str, Any] = {
        "planned": len(nodes),
        "started": sum(node.started_seq is not None for node in nodes),
    }
    for status in CodeNodeStatus:
        if status in {CodeNodeStatus.PLANNED, CodeNodeStatus.STARTED}:
            continue
        summary[status.value] = sum(node.status is status for node in nodes)
    roots: list[str] = []
    for node in nodes:
        candidates = node.root_failures or (
            (node.node_id,)
            if node.status not in {
                CodeNodeStatus.PLANNED,
                CodeNodeStatus.STARTED,
                CodeNodeStatus.SUCCEEDED,
                CodeNodeStatus.NOT_EXECUTED,
            }
            else ()
        )
        for candidate in candidates:
            if candidate not in roots:
                roots.append(candidate)
    summary["root_failures"] = roots
    return summary


def project_code_graph(
    state: Any, run_id: str, *, now: datetime | None = None
) -> CodeGraphView:
    """Project one run into a deterministic, presentation-neutral graph."""

    run = state.code_runs[run_id]
    nodes = sorted(
        (state.code_nodes[node_id] for node_id in run.node_ids),
        key=lambda node: node.ordinal,
    )
    ordinal_by_id = {node.node_id: node.ordinal for node in nodes}
    dependents: dict[str, list[int]] = {node.node_id: [] for node in nodes}
    for node in nodes:
        for dependency in node.dependencies:
            dependents[dependency].append(node.ordinal)
    event_times = {event.seq: _parse_timestamp(event.timestamp) for event in state.events}
    fallback_end = now or (
        _parse_timestamp(state.last_event_timestamp)
        if state.last_event_timestamp is not None
        else None
    )
    views = tuple(
        CodeGraphNodeView(
            node_id=node.node_id,
            ordinal=node.ordinal,
            name=node.name,
            target=_node_target(node.name, node.arguments),
            dependency_ordinals=tuple(
                ordinal_by_id[dependency] for dependency in node.dependencies
            ),
            dependent_ordinals=tuple(sorted(dependents[node.node_id])),
            status=node.status.value,
            is_current=node.status is CodeNodeStatus.STARTED,
            elapsed_ms=_elapsed_ms(
                event_times.get(node.started_seq),
                event_times.get(node.finished_seq) or fallback_end,
            ),
            result=node.result,
            blocked_by_ordinals=tuple(
                ordinal_by_id[item]
                for item in node.blocked_by
                if item in ordinal_by_id
            ),
            root_failure_ordinals=tuple(
                ordinal_by_id[item]
                for item in node.root_failures
                if item in ordinal_by_id
            ),
            approval=_approval_state(state.tool_calls[node.node_id]),
        )
        for node in nodes
    )
    run_events = [
        event
        for event in state.events
        if event.type in {"code_run_started", "code_run_finished"}
        and event.payload.get("run_id") == run_id
    ]
    started_at = next(
        (_parse_timestamp(event.timestamp) for event in run_events if event.type == "code_run_started"),
        None,
    )
    finished_at = next(
        (_parse_timestamp(event.timestamp) for event in reversed(run_events) if event.type == "code_run_finished"),
        None,
    )
    return CodeGraphView(
        run_id=run_id,
        description=run.description,
        status=run.status.value,
        nodes=views,
        summary=graph_summary(state, run_id),
        elapsed_ms=_elapsed_ms(started_at, finished_at or fallback_end),
        shell_mutation_warning=_has_unordered_shell_mutation(nodes),
    )


def _node_target(name: str, raw_arguments: str) -> str:
    try:
        arguments = json.loads(raw_arguments)
    except (TypeError, json.JSONDecodeError):
        return ""
    if not isinstance(arguments, dict):
        return ""
    if name == "bash":
        value = arguments.get("command")
    elif name == "grep":
        pattern = arguments.get("pattern")
        path = arguments.get("path")
        if isinstance(pattern, str) and isinstance(path, str):
            return f"{pattern} in {path}"
        value = pattern if isinstance(pattern, str) else path
    else:
        value = arguments.get("path")
    return value if isinstance(value, str) else ""


def _approval_state(call: Any) -> str | None:
    if call.name not in {"write_file", "edit_file", "bash"}:
        return None
    if call.approved is True:
        return "approved"
    if call.approved is False:
        return "denied"
    status = getattr(call.status, "value", call.status)
    return "waiting" if status == "requested" else None


def _parse_timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _elapsed_ms(start: datetime | None, end: datetime | None) -> int | None:
    if start is None or end is None:
        return None
    return max(0, round((end - start).total_seconds() * 1000))


def _has_unordered_shell_mutation(nodes: list[CodeNode]) -> bool:
    shells = [node for node in nodes if node.name == "bash"]
    mutations = [node for node in nodes if node.name in {"write_file", "edit_file"}]
    return any(
        _execution_intervals_overlap(shell, mutation)
        for shell in shells
        for mutation in mutations
    )


def _execution_intervals_overlap(first: CodeNode, second: CodeNode) -> bool:
    if first.started_seq is None or second.started_seq is None:
        return False
    first_end = first.finished_seq if first.finished_seq is not None else float("inf")
    second_end = second.finished_seq if second.finished_seq is not None else float("inf")
    return first.started_seq < second_end and second.started_seq < first_end


__all__ = [
    "CodeGraphNodeView",
    "CodeGraphView",
    "CodeNode",
    "CodeNodeStatus",
    "CodeRun",
    "CodeRunStatus",
    "graph_summary",
    "project_code_graph",
]
