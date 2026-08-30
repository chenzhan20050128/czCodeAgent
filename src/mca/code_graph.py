"""Pure values and summaries for durable Code Mode execution graphs."""

from __future__ import annotations

from dataclasses import dataclass, field
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


__all__ = ["CodeNode", "CodeNodeStatus", "CodeRun", "CodeRunStatus", "graph_summary"]
