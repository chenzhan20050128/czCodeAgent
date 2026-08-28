"""Project durable session facts into OpenAI-compatible chat messages.

The rollout remains the history source of truth.  Values that can change
between processes, such as the current working directory and Git status, are
accepted separately as live input for every projection.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date as Date
from typing import Any

from .conversation import ProjectionError, validate_conversation
from .domain import (
    DomainError,
    Event,
    SessionReducer,
    SessionState,
    ToolStatus,
)


class ProjectionBlockedError(ProjectionError):
    """Raised when an unknown side-effect outcome requires reconciliation."""


@dataclass(frozen=True)
class ProjectionEnvironment:
    """Current process facts supplied independently from the rollout."""

    cwd: str
    platform: str
    date: str
    is_git: bool

    def __post_init__(self) -> None:
        if not isinstance(self.cwd, str) or not self.cwd:
            raise ProjectionError("environment cwd must be a non-empty string")
        if not isinstance(self.platform, str) or not self.platform:
            raise ProjectionError(
                "environment platform must be a non-empty string"
            )
        if not isinstance(self.date, str) or not self.date:
            raise ProjectionError("environment date must be a non-empty string")
        if type(self.is_git) is not bool:
            raise ProjectionError("environment is_git must be a boolean")


_ENVIRONMENT_FIELDS = frozenset({"cwd", "platform", "date", "is_git"})
@dataclass(frozen=True)
class _CallOccurrence:
    call_key: str
    provider_call_id: str


class _EventCallTracker:
    """Resolve call references against their state at each event boundary."""

    def __init__(self) -> None:
        self._by_key: dict[str, _CallOccurrence] = {}
        self._by_provider_id: dict[str, _CallOccurrence] = {}

    def add_assistant_calls(
        self, event: Event, calls: Sequence[Mapping[str, Any]]
    ) -> None:
        for call in calls:
            provider_call_id = call["id"]
            if provider_call_id in self._by_provider_id:
                raise ProjectionError(
                    f"tool call ID is already unresolved: {provider_call_id!r}"
                )
            occurrence = _CallOccurrence(
                call_key=f"{event.seq}:{provider_call_id}",
                provider_call_id=provider_call_id,
            )
            self._by_key[occurrence.call_key] = occurrence
            self._by_provider_id[provider_call_id] = occurrence

    def resolve(self, event: Event) -> _CallOccurrence:
        call_key = event.payload.get("call_key")
        provider_call_id = event.payload.get("call_id")
        if call_key is not None:
            if not isinstance(call_key, str) or not call_key:
                raise ProjectionError(
                    "tool event call_key must be a non-empty string"
                )
            occurrence = self._by_key.get(call_key)
            if occurrence is None:
                raise ProjectionError(f"tool event has no active call: {call_key}")
            if (
                provider_call_id is not None
                and provider_call_id != occurrence.provider_call_id
            ):
                raise ProjectionError(
                    "tool event call_key and call_id identify different calls"
                )
            return occurrence

        if not isinstance(provider_call_id, str) or not provider_call_id:
            raise ProjectionError(
                "tool event must identify a provider call ID or call_key"
            )
        occurrence = self._by_provider_id.get(provider_call_id)
        if occurrence is None:
            raise ProjectionError(
                f"tool event has no active call ID: {provider_call_id!r}"
            )
        return occurrence

    def close(self, occurrence: _CallOccurrence) -> None:
        self._by_key.pop(occurrence.call_key, None)
        self._by_provider_id.pop(occurrence.provider_call_id, None)


def _plain_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _plain_json(item) for key, item in value.items()}
    if isinstance(value, tuple) or isinstance(value, list):
        return [_plain_json(item) for item in value]
    return value


def _environment(value: ProjectionEnvironment | Mapping[str, Any]) -> ProjectionEnvironment:
    if isinstance(value, ProjectionEnvironment):
        return value
    if not isinstance(value, Mapping):
        raise ProjectionError(
            "environment must be a ProjectionEnvironment or exact field mapping"
        )
    fields = set(value)
    if fields != _ENVIRONMENT_FIELDS:
        missing = sorted(_ENVIRONMENT_FIELDS - fields)
        extra = sorted(fields - _ENVIRONMENT_FIELDS)
        raise ProjectionError(
            f"environment fields mismatch (missing={missing}, extra={extra})"
        )
    raw_date = value["date"]
    if isinstance(raw_date, Date):
        raw_date = raw_date.isoformat()
    return ProjectionEnvironment(
        cwd=value["cwd"],
        platform=value["platform"],
        date=raw_date,
        is_git=value["is_git"],
    )


def _system_message(
    environment: ProjectionEnvironment, *, checkpoint_summary: str | None
) -> dict[str, str]:
    live_environment = json.dumps(
        {
            "cwd": environment.cwd,
            "platform": environment.platform,
            "date": environment.date,
            "is_git": environment.is_git,
        },
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    content = (
        "You are a coding assistant. You propose tool calls; the local MCA "
        "runtime validates and executes them. Do not claim that you executed "
        "a tool yourself. Never reveal or persist secrets.\n"
        "Current live environment (supplied for this request, not recovered "
        f"from the rollout): {live_environment}"
    )
    if checkpoint_summary:
        content += f"\nCompacted conversation summary:\n{checkpoint_summary}"
    return {"role": "system", "content": content}


def _normalize_tool_call(document: object) -> dict[str, Any]:
    if not isinstance(document, Mapping):
        raise ProjectionError("assistant tool call must be an object")
    call_id = document.get("id", document.get("call_id"))
    if not isinstance(call_id, str) or not call_id:
        raise ProjectionError("assistant tool call id must be a non-empty string")
    if document.get("type", "function") != "function":
        raise ProjectionError("only function tool calls can be projected")

    function = document.get("function")
    if function is not None:
        if not isinstance(function, Mapping):
            raise ProjectionError("assistant tool function must be an object")
        name = function.get("name")
        arguments = function.get("arguments")
    else:
        name = document.get("name")
        arguments = document.get("arguments")
    if not isinstance(name, str) or not name:
        raise ProjectionError("assistant tool name must be a non-empty string")
    if not isinstance(arguments, str):
        try:
            arguments = json.dumps(
                _plain_json(arguments),
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            )
        except (TypeError, ValueError):
            raise ProjectionError(
                "assistant tool arguments must be JSON-compatible"
            ) from None
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }


def _assistant_message(event: Event) -> dict[str, Any]:
    content = event.payload.get("content")
    if content is not None and not isinstance(content, str):
        raise ProjectionError("assistant content must be a string or null")
    reasoning_present = "reasoning_content" in event.payload
    reasoning_content = event.payload.get("reasoning_content")
    if reasoning_present and not isinstance(reasoning_content, str):
        raise ProjectionError(
            "assistant reasoning_content must be a string"
        )
    raw_calls = event.payload.get("tool_calls", ())
    if not isinstance(raw_calls, (tuple, list)):
        raise ProjectionError("assistant tool_calls must be an array")
    calls = [_normalize_tool_call(document) for document in raw_calls]
    if content is None and not calls:
        raise ProjectionError(
            "assistant fact must contain text or at least one tool call"
        )

    message: dict[str, Any] = {"role": "assistant", "content": content}
    if reasoning_present:
        message["reasoning_content"] = reasoning_content
    if calls:
        message["tool_calls"] = calls
    return message


def _tool_result_message(
    event: Event, occurrence: _CallOccurrence
) -> dict[str, str] | None:
    raw_status = event.payload.get("status")
    aliases = {"success": "succeeded", "timeout": "timed_out"}
    try:
        status = ToolStatus(aliases.get(raw_status, raw_status))
    except (TypeError, ValueError):
        raise ProjectionError(f"unknown projected tool status: {raw_status!r}") from None

    if status is ToolStatus.OUTCOME_UNKNOWN:
        return None
    if status in {ToolStatus.REQUESTED, ToolStatus.STARTED}:
        raise ProjectionError("tool_finished does not contain a terminal result")
    result = event.payload.get("result")
    if result is None:
        result = ""
    if not isinstance(result, str):
        raise ProjectionError("tool result content must be a string or null")
    return {
        "role": "tool",
        "tool_call_id": occurrence.provider_call_id,
        "content": result,
    }


def _reconciled_result_message(
    event: Event, occurrence: _CallOccurrence
) -> dict[str, str]:
    descriptions = {
        "succeeded": "User confirmed after recovery that the tool succeeded.",
        "user_confirmed_success": "User confirmed after recovery that the tool succeeded.",
        "failed": "User confirmed after recovery that the tool failed.",
        "user_confirmed_failure": "User confirmed after recovery that the tool failed.",
        "abandoned": "User abandoned the uncertain tool call after recovery.",
    }
    outcome = event.payload.get("outcome")
    content = descriptions.get(outcome)
    if content is None:
        raise ProjectionError(f"unknown reconciliation outcome: {outcome!r}")
    note = event.payload.get("note", "")
    if not isinstance(note, str):
        raise ProjectionError("reconciliation note must be a string")
    if note:
        content += f" Note: {note}"
    return {
        "role": "tool",
        "tool_call_id": occurrence.provider_call_id,
        "content": content,
    }


def _project_event(
    event: Event, call_tracker: _EventCallTracker
) -> dict[str, Any] | None:
    if event.type == "turn_started":
        user_input = event.payload.get("user_input", event.payload.get("input", ""))
        if not isinstance(user_input, str):
            raise ProjectionError("turn input must be a string")
        return {"role": "user", "content": user_input}
    if event.type == "assistant_accepted":
        message = _assistant_message(event)
        call_tracker.add_assistant_calls(event, message.get("tool_calls", ()))
        return message
    if event.type == "tool_finished":
        occurrence = call_tracker.resolve(event)
        message = _tool_result_message(event, occurrence)
        if message is not None:
            call_tracker.close(occurrence)
        return message
    if event.type == "tool_reconciled":
        occurrence = call_tracker.resolve(event)
        message = _reconciled_result_message(event, occurrence)
        call_tracker.close(occurrence)
        return message
    return None


class PromptProjector:
    """Pure projection of ordered rollout events plus current environment."""

    @staticmethod
    def project(
        events: Iterable[Event],
        state: SessionState,
        environment: ProjectionEnvironment | Mapping[str, Any],
    ) -> list[dict[str, Any]]:
        if not isinstance(state, SessionState):
            raise ProjectionError("state must be a SessionState")
        ordered_events = tuple(events)
        if any(not isinstance(event, Event) for event in ordered_events):
            raise ProjectionError("events must contain only Event values")
        try:
            replayed = SessionReducer.replay(ordered_events)
        except DomainError as exc:
            raise ProjectionError(f"invalid rollout facts: {exc}") from exc
        if replayed != state:
            raise ProjectionError("state does not match the supplied rollout facts")

        if state.pending_recovery_intent is not None:
            raise ProjectionBlockedError(
                "session has a pending recovery intent"
            )
        if state.recovery_blocked or any(
            call.status is ToolStatus.OUTCOME_UNKNOWN
            for call in state.tool_calls.values()
        ):
            raise ProjectionBlockedError(
                "session recovery is blocked by an unknown tool outcome"
            )
        unresolved = [
            call.call_key
            for call in state.tool_calls.values()
            if call.status in {ToolStatus.REQUESTED, ToolStatus.STARTED}
        ]
        if unresolved:
            raise ProjectionError(
                f"cannot project unresolved tool calls: {sorted(unresolved)}"
            )

        current_environment = _environment(environment)
        checkpoint: Event | None = None
        for event in ordered_events:
            if event.type == "compaction_checkpoint":
                checkpoint = event

        through_seq = 0
        summary: str | None = None
        baseline: list[dict[str, Any]] = []
        if checkpoint is not None:
            raw_through_seq = checkpoint.payload.get("through_seq")
            if (
                type(raw_through_seq) is not int
                or raw_through_seq < 0
                or raw_through_seq >= checkpoint.seq
            ):
                raise ProjectionError(
                    "checkpoint through_seq must reference an earlier event"
                )
            raw_summary = checkpoint.payload.get("summary")
            if not isinstance(raw_summary, str):
                raise ProjectionError("checkpoint summary must be a string")
            replacement = checkpoint.payload.get("replacement_conversation")
            if not isinstance(replacement, (list, tuple)):
                raise ProjectionError(
                    "checkpoint replacement_conversation must be an array"
                )
            through_seq = raw_through_seq
            summary = raw_summary
            baseline = _plain_json(replacement)

        messages: list[dict[str, Any]] = [
            _system_message(current_environment, checkpoint_summary=summary)
        ]
        messages.extend(baseline)
        call_tracker = _EventCallTracker()
        for event in ordered_events:
            projected = _project_event(event, call_tracker)
            if event.seq > through_seq and projected is not None:
                messages.append(projected)

        validate_conversation(messages)
        return messages


def _serialized_request(
    messages: Sequence[Mapping[str, Any]],
    tool_schemas: Sequence[Mapping[str, Any]] | None,
) -> str:
    request: dict[str, Any] = {"messages": _plain_json(messages)}
    if tool_schemas:
        request["tools"] = _plain_json(tool_schemas)
    try:
        return json.dumps(
            request,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError):
        raise ProjectionError(
            "messages and tool schemas must be safely JSON serializable"
        ) from None


def estimate_request_tokens(
    messages: Sequence[Mapping[str, Any]],
    tool_schemas: Sequence[Mapping[str, Any]] | None = None,
) -> int:
    """Return a deterministic, conservative tokenizer-independent estimate.

    All serialized request text is counted.  ASCII uses four characters per
    estimated token, while each non-ASCII character counts as one token.
    Eight tokens per message account for role/framing overhead.
    """

    if not isinstance(messages, (list, tuple)):
        raise ProjectionError("messages must be an array")
    if tool_schemas is not None and not isinstance(tool_schemas, (list, tuple)):
        raise ProjectionError("tool schemas must be an array or null")
    serialized = _serialized_request(messages, tool_schemas)
    ascii_characters = sum(ord(character) < 128 for character in serialized)
    non_ascii_characters = len(serialized) - ascii_characters
    return (
        non_ascii_characters
        + math.ceil(ascii_characters / 4)
        + len(messages) * 8
    )


def request_fits_budget(
    messages: Sequence[Mapping[str, Any]],
    tool_schemas: Sequence[Mapping[str, Any]] | None = None,
    *,
    context_window: int,
    reserved_output_tokens: int,
    safety_margin: int,
    last_usage: tuple[int, int, int] | None = None,
) -> bool:
    """Return whether estimated input fits after output and safety reserves."""

    for name, value, allow_zero in (
        ("context_window", context_window, False),
        ("reserved_output_tokens", reserved_output_tokens, True),
        ("safety_margin", safety_margin, True),
    ):
        minimum = 0 if allow_zero else 1
        if type(value) is not int or value < minimum:
            qualifier = "non-negative" if allow_zero else "positive"
            raise ProjectionError(f"{name} must be a {qualifier} integer")
    available_input = context_window - reserved_output_tokens - safety_margin
    estimate = usage_anchored_request_tokens(
        messages, tool_schemas, last_usage=last_usage
    )
    return estimate <= available_input


def usage_anchored_request_tokens(
    messages: Sequence[Mapping[str, Any]],
    tool_schemas: Sequence[Mapping[str, Any]] | None = None,
    *,
    last_usage: tuple[int, int, int] | None = None,
) -> int:
    """Estimate request tokens, anchored on the last real provider usage.

    The provider's ``total_tokens`` already priced the system prompt, tool
    schemas, and history up to and including the last assistant reply.  Only
    the messages appended after that reply (tool results, a new user turn) are
    unpriced, so they are added with the heuristic.  The result is never below
    the pure heuristic, so anchoring can only trigger compaction earlier, never
    skip it.
    """

    heuristic = estimate_request_tokens(messages, tool_schemas)
    if last_usage is None:
        return heuristic
    if (
        not isinstance(last_usage, tuple)
        or len(last_usage) != 3
        or any(type(item) is not int or item < 0 for item in last_usage)
    ):
        raise ProjectionError(
            "last_usage must be three non-negative integers or None"
        )
    tail = _messages_after_last_assistant(messages)
    tail_schemas = tool_schemas if tail is messages else None
    anchored = last_usage[2] + estimate_request_tokens(tail, tail_schemas)
    return max(heuristic, anchored)


def _messages_after_last_assistant(
    messages: Sequence[Mapping[str, Any]],
) -> Sequence[Mapping[str, Any]]:
    for index in range(len(messages) - 1, -1, -1):
        if messages[index].get("role") == "assistant":
            return messages[index + 1 :]
    return messages
