"""Bounded turn orchestration over durable session facts."""

from __future__ import annotations

import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from .config import Config
from .domain import (
    Event,
    SamplingOutcome,
    SessionReducer,
    SessionState,
    ToolCall,
    ToolStatus,
    TurnStatus,
)
from .executor import AcceptedToolCall, ToolExecutor, ToolExecutorError
from .model import SamplingResult
from .projection import ProjectionEnvironment, PromptProjector
from .store import RolloutStore
from .tools.registry import ToolRegistry


MAX_STEPS_FALLBACK = "Maximum tool steps reached before completion."

_FAILURE_MESSAGES = {
    SamplingOutcome.LENGTH_EXCEEDED: (
        TurnStatus.FAILED,
        "model output exceeded the configured length limit",
    ),
    SamplingOutcome.FILTERED: (
        TurnStatus.FAILED,
        "model response was filtered",
    ),
    SamplingOutcome.PROTOCOL_ERROR: (
        TurnStatus.FAILED,
        "model returned an invalid response",
    ),
    SamplingOutcome.TRANSPORT_INTERRUPTED: (
        TurnStatus.FAILED,
        "model transport was interrupted",
    ),
    SamplingOutcome.ABORTED: (
        TurnStatus.INTERRUPTED,
        "model sampling was aborted",
    ),
}


class AgentLoopError(RuntimeError):
    """Raised when the loop cannot safely continue."""


class RecoveryBlockedError(AgentLoopError):
    """Raised when an uncertain tool outcome must be reconciled first."""


class ModelSampler(Protocol):
    def sample(
        self,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]],
        allow_tools: bool,
        *,
        on_content: Callable[[str], None] | None = None,
        on_invalidate: Callable[[], None] | None = None,
    ) -> SamplingResult: ...


class Compactor(Protocol):
    def __call__(self) -> object: ...


@dataclass(frozen=True)
class TurnResult:
    """Stable result returned after one handled turn exit."""

    status: TurnStatus
    final_text: str
    turn_id: str
    tool_steps: int
    error: str | None


class AgentLoop:
    """Own one active turn and drive sampling and tools to a terminal fact."""

    def __init__(
        self,
        *,
        config: Config,
        store: RolloutStore,
        state: SessionState,
        model: ModelSampler,
        executor: ToolExecutor,
        environment: Callable[[], ProjectionEnvironment | Mapping[str, Any]],
        compactor: Compactor | None = None,
        on_content: Callable[[str], None] | None = None,
        on_invalidate: Callable[[], None] | None = None,
    ) -> None:
        if config.max_steps < 0:
            raise ValueError("max_steps must be non-negative")
        if config.max_tool_calls_per_batch < 1:
            raise ValueError("max_tool_calls_per_batch must be positive")
        if executor.store is not store or executor.state is not state:
            raise ValueError("executor must share the AgentLoop store and state")
        if state.session_id != store.session_id:
            raise ValueError("state and store must belong to the same session")

        self.config = config
        self.store = store
        self.state = state
        self.model = model
        self.executor = executor
        self.registry: ToolRegistry = executor.registry
        self.environment = environment
        self.compactor = compactor
        self.on_content = on_content
        self.on_invalidate = on_invalidate
        self._usable = True

    def run_turn(self, user_input: str) -> TurnResult:
        """Run one user turn without maintaining a second message history."""

        self._require_ready(user_input)
        turn_id = str(uuid.uuid4())
        self._append(
            "turn_started", {"turn_id": turn_id, "user_input": user_input}
        )
        tool_steps = 0

        try:
            while tool_steps < self.config.max_steps:
                sampled = self._sample_with_one_compaction(allow_tools=True)
                if sampled is None:
                    return self._finish(
                        turn_id,
                        TurnStatus.FAILED,
                        tool_steps,
                        error="context compaction failed",
                    )
                if sampled.outcome is SamplingOutcome.CONTEXT_OVERFLOW:
                    return self._finish(
                        turn_id,
                        TurnStatus.FAILED,
                        tool_steps,
                        error="model context overflow persisted after compaction",
                    )
                if sampled.outcome is SamplingOutcome.COMPLETE_TEXT:
                    if not sampled.content or sampled.tool_calls:
                        return self._finish(
                            turn_id,
                            TurnStatus.FAILED,
                            tool_steps,
                            error="model returned an empty text response",
                        )
                    self._accept_assistant(turn_id, sampled, include_calls=False)
                    return self._finish(
                        turn_id,
                        TurnStatus.COMPLETED,
                        tool_steps,
                        final_text=sampled.content,
                    )
                if sampled.outcome is SamplingOutcome.VALID_TOOL_BATCH:
                    if not sampled.tool_calls:
                        return self._finish(
                            turn_id,
                            TurnStatus.FAILED,
                            tool_steps,
                            error="model returned an invalid tool batch",
                        )
                    accepted = self._accept_assistant(
                        turn_id, sampled, include_calls=True
                    )
                    tool_steps += 1
                    interrupted = self._execute_batch(accepted)
                    if interrupted:
                        return self._finish(
                            turn_id,
                            TurnStatus.INTERRUPTED,
                            tool_steps,
                            error="turn interrupted by user",
                        )
                    continue
                terminal = _FAILURE_MESSAGES.get(sampled.outcome)
                if terminal is not None:
                    status, message = terminal
                    return self._finish(
                        turn_id, status, tool_steps, error=message
                    )
                return self._finish(
                    turn_id,
                    TurnStatus.FAILED,
                    tool_steps,
                    error="model returned an unsupported sampling outcome",
                )

            return self._finalize_after_limit(turn_id, tool_steps)
        except KeyboardInterrupt:
            self._close_open_calls_after_interrupt(turn_id)
            return self._finish(
                turn_id,
                TurnStatus.INTERRUPTED,
                tool_steps,
                error="turn interrupted by user",
            )

    def _require_ready(self, user_input: str) -> None:
        if not self._usable:
            raise AgentLoopError("agent loop is unusable after reducer divergence")
        if not isinstance(user_input, str):
            raise TypeError("user_input must be a string")
        if self.state.recovery_blocked:
            raise RecoveryBlockedError(
                "session recovery is blocked by an unknown tool outcome"
            )
        if self.state.active_turn_id is not None:
            raise AgentLoopError("session already has an active turn")
        if self.state.session_id is None:
            raise AgentLoopError("session has not been created")

    def _sample(self, *, allow_tools: bool) -> SamplingResult:
        messages = PromptProjector.project(
            self.store.load(), self.state, self.environment()
        )
        schemas = self.registry.provider_schemas() if allow_tools else []
        return self.model.sample(
            messages,
            schemas,
            allow_tools,
            on_content=self.on_content,
            on_invalidate=self.on_invalidate,
        )

    def _sample_with_one_compaction(
        self, *, allow_tools: bool
    ) -> SamplingResult | None:
        sampled = self._sample(allow_tools=allow_tools)
        if sampled.outcome is not SamplingOutcome.CONTEXT_OVERFLOW:
            return sampled
        if self.compactor is None:
            return None
        try:
            compacted = self.compactor()
        except Exception:
            return None
        if not compacted:
            return None
        return self._sample(allow_tools=allow_tools)

    def _accept_assistant(
        self, turn_id: str, sampled: SamplingResult, *, include_calls: bool
    ) -> tuple[AcceptedToolCall, ...]:
        call_documents = (
            [
                {
                    "id": call.id,
                    "type": call.type,
                    "function": {
                        "name": call.name,
                        "arguments": call.arguments,
                    },
                }
                for call in sampled.tool_calls
            ]
            if include_calls
            else []
        )
        event = self._append(
            "assistant_accepted",
            {
                "turn_id": turn_id,
                "content": sampled.content or None,
                "reasoning_content": sampled.reasoning_content,
                "finish_reason": sampled.finish_reason,
                "tool_calls": call_documents,
            },
        )
        return tuple(
            AcceptedToolCall.from_tool_call(
                self.state.tool_calls[f"{event.seq}:{call.id}"]
            )
            for call in sampled.tool_calls
        ) if include_calls else ()

    def _execute_batch(self, calls: Sequence[AcceptedToolCall]) -> bool:
        limit = self.config.max_tool_calls_per_batch
        for index, call in enumerate(calls[:limit]):
            try:
                result = self.executor.execute(call)
            except KeyboardInterrupt:
                self._close_calls(
                    calls[index:],
                    current_call_key=call.call_key,
                    current_status=ToolStatus.CANCELLED,
                )
                return True
            except ToolExecutorError as error:
                self._usable = False
                raise AgentLoopError("tool executor cannot safely continue") from error
            if result.status == ToolStatus.INTERRUPTED.value:
                self._close_calls(calls[index + 1 :])
                return True

        for call in calls[limit:]:
            self._finish_requested_call(
                call,
                ToolStatus.BATCH_LIMIT_EXCEEDED,
                "tool call skipped: batch limit exceeded",
            )
        return False

    def _close_open_calls_after_interrupt(self, turn_id: str) -> None:
        open_calls = [
            AcceptedToolCall.from_tool_call(call)
            for call in self.state.tool_calls.values()
            if call.turn_id == turn_id
            and call.status in {ToolStatus.REQUESTED, ToolStatus.STARTED}
        ]
        self._close_calls(open_calls)

    def _close_calls(
        self,
        calls: Sequence[AcceptedToolCall],
        *,
        current_call_key: str | None = None,
        current_status: ToolStatus = ToolStatus.CANCELLED,
    ) -> None:
        for call in calls:
            state_call = self.state.tool_calls[call.call_key]
            if state_call.is_terminal:
                continue
            status = (
                current_status
                if call.call_key == current_call_key
                else ToolStatus.NOT_EXECUTED
            )
            if state_call.status is ToolStatus.STARTED:
                status = ToolStatus.CANCELLED
            self._finish_requested_call(
                call, status, "tool call not executed because the turn was interrupted"
            )

    def _finish_requested_call(
        self, call: AcceptedToolCall, status: ToolStatus, result: str
    ) -> None:
        self._append(
            "tool_finished",
            {
                "call_key": call.call_key,
                "call_id": call.provider_call_id,
                "status": status.value,
                "result": result,
                "truncated": False,
            },
        )

    def _finalize_after_limit(self, turn_id: str, tool_steps: int) -> TurnResult:
        try:
            sampled = self._sample(allow_tools=False)
        except KeyboardInterrupt:
            return self._finish(
                turn_id,
                TurnStatus.INTERRUPTED,
                tool_steps,
                error="turn interrupted by user",
            )
        final_text = MAX_STEPS_FALLBACK
        if (
            sampled.outcome is SamplingOutcome.COMPLETE_TEXT
            and bool(sampled.content)
            and not sampled.tool_calls
        ):
            self._accept_assistant(turn_id, sampled, include_calls=False)
            final_text = sampled.content
        return self._finish(
            turn_id,
            TurnStatus.MAX_STEPS_REACHED,
            tool_steps,
            final_text=final_text,
            error="maximum tool steps reached",
        )

    def _finish(
        self,
        turn_id: str,
        status: TurnStatus,
        tool_steps: int,
        *,
        final_text: str = "",
        error: str | None = None,
    ) -> TurnResult:
        payload: dict[str, Any] = {
            "turn_id": turn_id,
            "status": status.value,
        }
        if final_text:
            payload["final_text"] = final_text
        if error is not None:
            payload["error"] = error
        self._append("turn_finished", payload)
        return TurnResult(status, final_text, turn_id, tool_steps, error)

    def _append(self, event_type: str, payload: Mapping[str, Any]) -> Event:
        if not self._usable:
            raise AgentLoopError("agent loop is unusable after reducer divergence")
        event = self.store.append(event_type, payload)
        try:
            SessionReducer.apply(self.state, event)
        except Exception as error:
            self._usable = False
            raise AgentLoopError(
                f"durable event {event.seq} could not be applied to state"
            ) from error
        return event


__all__ = [
    "AgentLoop",
    "AgentLoopError",
    "Compactor",
    "MAX_STEPS_FALLBACK",
    "ModelSampler",
    "RecoveryBlockedError",
    "TurnResult",
]
