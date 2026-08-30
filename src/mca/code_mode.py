"""Outer run_code orchestration over the constrained Python runtime."""

from __future__ import annotations

import hashlib
import json
import uuid
import threading
from dataclasses import dataclass

from .code_graph import CodeRunStatus, graph_summary
from .code_runtime import CodeRuntime, CodeRuntimeConfig
from .code_scheduler import CodeDagScheduler
from .domain import SessionReducer, SessionState, ToolCall
from .executor import AcceptedToolCall, ToolExecutor
from .store import RolloutStore
from .tools.registry import ToolResult


@dataclass(frozen=True)
class PreparedCodeProgram:
    description: str
    code: str


def prepare_code_program(arguments: dict[str, object]) -> PreparedCodeProgram:
    description = arguments.get("description")
    code = arguments.get("code")
    if not isinstance(description, str) or not description.strip():
        raise ValueError("description must be a non-empty string")
    if not isinstance(code, str) or not code.strip():
        raise ValueError("code must be a non-empty string")
    return PreparedCodeProgram(description.strip(), code)


class CodeModeRunner:
    """Execute one outer run_code call and persist its nested graph."""

    def __init__(
        self,
        *,
        store: RolloutStore,
        state: SessionState,
        executor: ToolExecutor,
        runtime: CodeRuntime | None = None,
        max_parallel: int = 4,
        max_nodes: int = 64,
    ) -> None:
        self.store = store
        self.state = state
        self.executor = executor
        self.runtime = runtime or CodeRuntime(CodeRuntimeConfig())
        self.max_parallel = max_parallel
        self.max_nodes = max_nodes

    def run(
        self,
        outer: AcceptedToolCall | ToolCall,
        *,
        description: str,
        code: str,
    ) -> ToolResult:
        call = (
            outer if isinstance(outer, ToolCall) else self.state.tool_calls[outer.call_key]
        )
        run_id = str(uuid.uuid4())
        self._append(
            "code_run_started",
            {
                "run_id": run_id,
                "turn_id": call.turn_id,
                "parent_call_key": call.call_key,
                "description": description,
                "source_hash": hashlib.sha256(code.encode("utf-8")).hexdigest(),
            },
        )
        cancellation_event = threading.Event()
        scheduler = CodeDagScheduler(
            store=self.store,
            state=self.state,
            executor=self.executor,
            run_id=run_id,
            max_parallel=self.max_parallel,
            max_nodes=self.max_nodes,
            cancellation_event=cancellation_event,
        )
        runtime_result = self.runtime.run(
            code,
            execute_graph=scheduler.execute_graph,
            cancellation_event=cancellation_event,
        )
        summary = graph_summary(self.state, run_id)
        failed = any(
            summary.get(key, 0)
            for key in (
                "failed", "denied", "invalid_arguments", "unknown_tool",
                "conflict", "timed_out", "interrupted",
                "upstream_failed", "outcome_unknown", "abandoned",
            )
        )
        status = CodeRunStatus.FAILED if runtime_result.error or failed else CodeRunStatus.SUCCEEDED
        payload = {
            "result": runtime_result.value,
            "logs": list(runtime_result.logs),
            "execution_summary": summary,
        }
        if runtime_result.error is not None:
            payload["runtime_error"] = {
                "code": runtime_result.error.code,
                "message": runtime_result.error.message,
            }
        rendered = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        self._append(
            "code_run_finished",
            {
                "run_id": run_id,
                "status": status.value,
                "result": rendered,
                "summary": summary,
            },
        )
        return ToolResult.bounded(
            title=f"run_code: {description}",
            output=rendered,
            status=("failed" if status is CodeRunStatus.FAILED else "succeeded"),
        )

    def _append(self, event_type: str, payload: dict[str, object]) -> None:
        event = self.store.append(event_type, payload)
        SessionReducer.apply(self.state, event)
        self.executor.observe_event(event)


__all__ = ["CodeModeRunner", "PreparedCodeProgram", "prepare_code_program"]
