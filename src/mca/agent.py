"""Bounded turn orchestration over durable session facts."""

from __future__ import annotations

import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, fields
from typing import Any, Protocol

from .config import Config
from .domain import (
    DomainError,
    Event,
    SamplingOutcome,
    SessionReducer,
    SessionState,
    ToolCall,
    ToolStatus,
    TurnStatus,
    reduce_event,
)
from .executor import AcceptedToolCall, ToolExecutor, ToolExecutorError
from .model import SamplingResult
from .projection import (
    ProjectionEnvironment,
    PromptProjector,
    request_fits_budget,
)
from .store import RolloutStore
from .tool_scheduler import ToolBatchScheduler
from .tools.registry import ToolRegistry


MAX_STEPS_FALLBACK = "Maximum tool steps reached before completion."
REQUEST_SAFETY_MARGIN = 256

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
        on_reasoning: Callable[[str], None] | None = None,
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


class _StreamObservation:
    """Track whether one logical sample left unaccepted text on screen."""

    def __init__(
        self,
        on_content: Callable[[str], None] | None,
        on_invalidate: Callable[[], None] | None,
    ) -> None:
        self._on_content = on_content
        self._on_invalidate = on_invalidate
        self._needs_invalidation = False

    def content(self, delta: str) -> None:
        if delta:
            self._needs_invalidation = True
        if self._on_content is not None:
            try:
                self._on_content(delta)
            except Exception:
                pass

    def invalidate(self) -> None:
        self._needs_invalidation = False
        if self._on_invalidate is not None:
            try:
                self._on_invalidate()
            except Exception:
                pass

    def discard(self) -> None:
        if not self._needs_invalidation:
            return
        self._needs_invalidation = False
        if self._on_invalidate is not None:
            try:
                self._on_invalidate()
            except Exception:
                pass


@dataclass(frozen=True)
class _ObservedSample:
    result: SamplingResult
    stream: _StreamObservation

    def discard(self) -> None:
        self.stream.discard()


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
        on_reasoning: Callable[[str], None] | None = None,
        on_tool_calls: Callable[[Sequence[SampledToolCall]], None] | None = None,
    ) -> None:
        if config.max_steps < 0:
            raise ValueError("max_steps must be non-negative")
        if config.max_tool_calls_per_batch < 1:
            raise ValueError("max_tool_calls_per_batch must be positive")
        if config.max_parallel_tool_calls < 1:
            raise ValueError("max_parallel_tool_calls must be positive")
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
        self.on_reasoning = on_reasoning
        self.on_tool_calls = on_tool_calls
        self._usable = True

    def run_turn(self, user_input: str) -> TurnResult:
        """Run one user turn without maintaining a second message history."""

        self._require_ready(user_input)
        turn_id = str(uuid.uuid4())
        return self._drive_turn(
            turn_id, tool_steps=0, new_user_input=user_input
        )

    def resume_active_turn(self) -> TurnResult:
        """Continue the replayed active Turn after recovery reconciliation."""

        if not self._usable:
            raise AgentLoopError("agent loop is unusable after reducer divergence")
        if self.state.session_id is None:
            raise AgentLoopError("session has not been created")
        if self.state.recovery_blocked:
            raise RecoveryBlockedError(
                "session recovery is blocked by an unknown tool outcome"
            )
        turn_id = self.state.active_turn_id
        if turn_id is None or self.state.turns.get(turn_id) is not TurnStatus.ACTIVE:
            raise AgentLoopError("session has no active turn to resume")
        unresolved = [
            call.call_key
            for call in self.state.tool_calls.values()
            if call.turn_id == turn_id
            and call.status
            in {
                ToolStatus.REQUESTED,
                ToolStatus.STARTED,
                ToolStatus.OUTCOME_UNKNOWN,
            }
        ]
        if unresolved:
            raise RecoveryBlockedError(
                "active turn has unresolved recovery tool calls"
            )
        turn_started_seq = max(
            event.seq
            for event in self.state.events
            if event.type == "turn_started"
            and event.payload.get("turn_id") == turn_id
        )
        tool_steps = sum(
            event.type == "assistant_accepted"
            and event.seq > turn_started_seq
            and bool(event.payload.get("tool_calls", ()))
            for event in self.state.events
        )
        return self._drive_turn(
            turn_id, tool_steps=tool_steps, new_user_input=None
        )

    def _drive_turn(
        self,
        turn_id: str,
        *,
        tool_steps: int,
        new_user_input: str | None,
    ) -> TurnResult:
        observed: _ObservedSample | None = None

        try:
            if new_user_input is not None:
                self._append(
                    "turn_started",
                    {"turn_id": turn_id, "user_input": new_user_input},
                )
            while tool_steps < self.config.max_steps:
                observed = self._sample_with_one_compaction(allow_tools=True)
                if observed is None:
                    return self._finish(
                        turn_id,
                        TurnStatus.FAILED,
                        tool_steps,
                        error="context compaction failed",
                    )
                sampled = observed.result
                if sampled.outcome is SamplingOutcome.CONTEXT_OVERFLOW:
                    observed.discard()
                    return self._finish(
                        turn_id,
                        TurnStatus.FAILED,
                        tool_steps,
                        error="model context overflow persisted after compaction",
                    )
                if sampled.outcome is SamplingOutcome.COMPLETE_TEXT:
                    if not sampled.content or sampled.tool_calls:
                        observed.discard()
                        return self._finish(
                            turn_id,
                            TurnStatus.FAILED,
                            tool_steps,
                            error="model returned an empty text response",
                        )
                    before_accept = self.state.last_seq
                    try:
                        self._accept_assistant(
                            turn_id, sampled, include_calls=False
                        )
                    except KeyboardInterrupt:
                        if self._accepted_assistant_event(
                            before_accept + 1, turn_id
                        ) is None:
                            raise
                    observed = None
                    return self._finish(
                        turn_id,
                        TurnStatus.COMPLETED,
                        tool_steps,
                        final_text=sampled.content,
                    )
                if sampled.outcome is SamplingOutcome.VALID_TOOL_BATCH:
                    if not sampled.tool_calls:
                        observed.discard()
                        return self._finish(
                            turn_id,
                            TurnStatus.FAILED,
                            tool_steps,
                            error="model returned an invalid tool batch",
                        )
                    if self.on_tool_calls is not None:
                        try:
                            self.on_tool_calls(sampled.tool_calls)
                        except Exception:
                            pass
                    before_accept = self.state.last_seq
                    try:
                        accepted = self._accept_assistant(
                            turn_id, sampled, include_calls=True
                        )
                    except KeyboardInterrupt:
                        accepted_event = self._accepted_assistant_event(
                            before_accept + 1, turn_id
                        )
                        if accepted_event is None:
                            raise
                        observed = None
                        tool_steps += 1
                        accepted = self._accepted_calls(accepted_event, sampled)
                        self._close_calls(accepted)
                        return self._finish(
                            turn_id,
                            TurnStatus.INTERRUPTED,
                            tool_steps,
                            error="turn interrupted by user",
                        )
                    observed = None
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
                    observed.discard()
                    status, message = terminal
                    return self._finish(
                        turn_id, status, tool_steps, error=message
                    )
                observed.discard()
                return self._finish(
                    turn_id,
                    TurnStatus.FAILED,
                    tool_steps,
                    error="model returned an unsupported sampling outcome",
                )

            return self._finalize_after_limit(turn_id, tool_steps)
        except KeyboardInterrupt:
            if observed is not None:
                observed.discard()
            persisted = self._persisted_turn_result(turn_id, tool_steps)
            if persisted is not None:
                return persisted
            if self.state.active_turn_id != turn_id:
                raise
            self._close_open_calls_after_interrupt(turn_id)
            return self._finish(
                turn_id,
                TurnStatus.INTERRUPTED,
                tool_steps,
                error="turn interrupted by user",
            )
        except AgentLoopError:
            raise
        except Exception:
            if observed is not None:
                observed.discard()
            self._close_open_calls_after_interrupt(turn_id)
            return self._finish(
                turn_id,
                TurnStatus.FAILED,
                tool_steps,
                error="agent turn failed",
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

    def _project_request(
        self, *, allow_tools: bool
    ) -> tuple[list[dict[str, Any]], list[Mapping[str, Any]]]:
        messages = PromptProjector.project(
            self.store.load(), self.state, self.environment()
        )
        schemas = self.registry.provider_schemas() if allow_tools else []
        return messages, schemas

    def _sample_projected(
        self,
        messages: Sequence[Mapping[str, Any]],
        schemas: Sequence[Mapping[str, Any]],
        *,
        allow_tools: bool,
    ) -> _ObservedSample:
        stream = _StreamObservation(self.on_content, self.on_invalidate)
        try:
            sampled = self.model.sample(
                messages,
                schemas,
                allow_tools,
                on_content=stream.content,
                on_invalidate=stream.invalidate,
                on_reasoning=self.on_reasoning,
            )
        except KeyboardInterrupt:
            stream.discard()
            raise
        except Exception:
            stream.discard()
            raise
        return _ObservedSample(sampled, stream)

    def _sample(self, *, allow_tools: bool) -> _ObservedSample:
        messages, schemas = self._project_request(allow_tools=allow_tools)
        return self._sample_projected(
            messages, schemas, allow_tools=allow_tools
        )

    def _request_fits(
        self,
        messages: Sequence[Mapping[str, Any]],
        schemas: Sequence[Mapping[str, Any]],
    ) -> bool:
        return request_fits_budget(
            messages,
            schemas,
            context_window=self.config.context_window,
            reserved_output_tokens=self.config.request_max_output_tokens,
            safety_margin=REQUEST_SAFETY_MARGIN,
            last_usage=self.state.last_usage,
        )

    def _compact_once(self) -> bool:
        if self.compactor is None:
            return False
        try:
            return bool(self.compactor())
        except Exception:
            return False

    def _sample_with_one_compaction(
        self, *, allow_tools: bool
    ) -> _ObservedSample | None:
        messages, schemas = self._project_request(allow_tools=allow_tools)
        compacted = False
        if not self._request_fits(messages, schemas):
            if not self._compact_once():
                return None
            compacted = True
            messages, schemas = self._project_request(allow_tools=allow_tools)
            if not self._request_fits(messages, schemas):
                return None

        observed = self._sample_projected(
            messages, schemas, allow_tools=allow_tools
        )
        if observed.result.outcome is not SamplingOutcome.CONTEXT_OVERFLOW:
            return observed
        observed.discard()
        if compacted:
            return observed
        if not self._compact_once():
            return None
        messages, schemas = self._project_request(allow_tools=allow_tools)
        if not self._request_fits(messages, schemas):
            return None
        return self._sample_projected(
            messages, schemas, allow_tools=allow_tools
        )

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
                "usage": _usage_payload(sampled.usage),
            },
        )
        return self._accepted_calls(event, sampled) if include_calls else ()

    def _accepted_calls(
        self, event: Event, sampled: SamplingResult
    ) -> tuple[AcceptedToolCall, ...]:
        return tuple(
            AcceptedToolCall.from_tool_call(
                self.state.tool_calls[f"{event.seq}:{call.id}"]
            )
            for call in sampled.tool_calls
        )

    def _accepted_assistant_event(
        self, expected_seq: int, turn_id: str
    ) -> Event | None:
        if not self.state.assistant_events:
            return None
        event = self.state.assistant_events[-1]
        if (
            event.seq != expected_seq
            or event.payload.get("turn_id") != turn_id
        ):
            return None
        return event

    def _execute_batch(self, calls: Sequence[AcceptedToolCall]) -> bool:
        limit = self.config.max_tool_calls_per_batch
        scheduler = ToolBatchScheduler(
            self.executor,
            max_parallel=self.config.max_parallel_tool_calls,
            close_calls=self._close_calls,
        )
        try:
            interrupted = scheduler.execute(calls[:limit])
        except KeyboardInterrupt:
            self._synchronize_state_from_store()
            self._close_calls(calls[:limit])
            return True
        except ToolExecutorError as error:
            self._usable = False
            raise AgentLoopError("tool executor cannot safely continue") from error
        if interrupted:
            self._close_calls(calls[limit:])
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
        observed = self._sample_with_one_compaction(allow_tools=False)
        if observed is None:
            return self._finish(
                turn_id,
                TurnStatus.MAX_STEPS_REACHED,
                tool_steps,
                final_text=MAX_STEPS_FALLBACK,
                error="maximum tool steps reached; context compaction failed",
            )
        sampled = observed.result
        final_text = MAX_STEPS_FALLBACK
        if (
            sampled.outcome is SamplingOutcome.COMPLETE_TEXT
            and bool(sampled.content)
            and not sampled.tool_calls
        ):
            before_accept = self.state.last_seq
            try:
                self._accept_assistant(turn_id, sampled, include_calls=False)
            except KeyboardInterrupt:
                if self._accepted_assistant_event(
                    before_accept + 1, turn_id
                ) is None:
                    observed.discard()
                    raise
            except Exception:
                observed.discard()
                raise
            final_text = sampled.content
        else:
            observed.discard()
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
        persisted = self._persisted_turn_result(turn_id, tool_steps)
        if persisted is not None:
            return persisted
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

    def _persisted_turn_result(
        self, turn_id: str, tool_steps: int
    ) -> TurnResult | None:
        if self.state.active_turn_id == turn_id:
            return None
        status = self.state.turns.get(turn_id)
        if status is None or status in {TurnStatus.ACTIVE, TurnStatus.RECOVERY_BLOCKED}:
            return None
        terminal = next(
            (
                event
                for event in reversed(self.state.events)
                if event.type == "turn_finished"
                and event.payload.get("turn_id") == turn_id
            ),
            None,
        )
        if terminal is None:
            raise AgentLoopError("terminal turn has no durable finish event")
        final_text = terminal.payload.get("final_text", "")
        error = terminal.payload.get("error")
        if not isinstance(final_text, str) or (
            error is not None and not isinstance(error, str)
        ):
            raise AgentLoopError("terminal turn has an invalid durable result")
        return TurnResult(status, final_text, turn_id, tool_steps, error)

    def _append(self, event_type: str, payload: Mapping[str, Any]) -> Event:
        if not self._usable:
            raise AgentLoopError("agent loop is unusable after reducer divergence")
        candidate = Event.create(
            seq=self.state.last_seq + 1,
            session_id=self.store.session_id,
            event_type=event_type,
            payload=payload,
        )
        try:
            reduce_event(self.state, candidate)
        except DomainError:
            raise
        except Exception as error:
            self._usable = False
            raise AgentLoopError("candidate event could not be reduced") from error
        try:
            event = self.store.append(event_type, payload)
        except KeyboardInterrupt:
            self._synchronize_state_from_store()
            raise
        except Exception as error:
            self._usable = False
            raise AgentLoopError("rollout store append failed") from error
        try:
            SessionReducer.apply(self.state, event)
        except Exception as error:
            self._usable = False
            raise AgentLoopError(
                f"durable event {event.seq} could not be applied to state"
            ) from error
        return event

    def _synchronize_state_from_store(self) -> None:
        """Replay durable facts into the existing shared state object."""

        try:
            synchronized = SessionReducer.replay(self.store.load())
        except Exception as error:
            self._usable = False
            raise AgentLoopError(
                "rollout state could not be synchronized after interruption"
            ) from error
        for state_field in fields(SessionState):
            setattr(self.state, state_field.name, getattr(synchronized, state_field.name))


def _usage_payload(usage: object) -> dict[str, int] | None:
    """Convert an optional provider TokenUsage into a durable fact payload."""

    if usage is None:
        return None
    return {
        "prompt_tokens": usage.prompt_tokens,
        "completion_tokens": usage.completion_tokens,
        "total_tokens": usage.total_tokens,
    }


__all__ = [
    "AgentLoop",
    "AgentLoopError",
    "Compactor",
    "MAX_STEPS_FALLBACK",
    "ModelSampler",
    "RecoveryBlockedError",
    "REQUEST_SAFETY_MARGIN",
    "TurnResult",
]
