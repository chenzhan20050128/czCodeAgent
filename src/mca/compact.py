"""Conversation compaction without mutating the durable fact history."""

from __future__ import annotations

import copy
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from .domain import (
    DomainError,
    Event,
    SamplingOutcome,
    SessionReducer,
    SessionState,
    reduce_event,
)
from .model import SamplingResult
from .projection import (
    ProjectionEnvironment,
    ProjectionError,
    PromptProjector,
    validate_conversation,
)
from .store import RolloutStore


DEFAULT_TAIL_GROUP_COUNT = 4
DEFAULT_MAX_SUMMARY_INPUT_CHARS = 48_000
DEFAULT_MAX_OLD_TOOL_CHARS = 2_000
SUMMARY_SECTIONS = (
    "Goal",
    "Completed",
    "Decisions",
    "Constraints",
    "Workspace state",
    "Next steps",
    "Uncertainties",
)
SUMMARY_PROMPT = (
    "Summarize the supplied coding-agent conversation as a precise handoff. "
    "Return exactly these Markdown sections in this order, using the section "
    "names verbatim: "
    + ", ".join(f"## {section}" for section in SUMMARY_SECTIONS)
    + ". Preserve concrete paths, commands, observed results, remaining work, "
    "and uncertainty. Do not invent facts or tool results."
)
_SHORTENING_MARKER = "[tool output shortened for compaction]"


class CompactionError(RuntimeError):
    """Raised when a safe checkpoint cannot be produced or persisted."""


class SummaryModel(Protocol):
    def sample(
        self,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]],
        allow_tools: bool,
        **kwargs: object,
    ) -> SamplingResult: ...


@dataclass(frozen=True)
class AtomicGroup:
    """One indivisible provider-protocol group in the visible history."""

    index: int
    messages: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class CompactionDraft:
    """Pure, validated inputs prepared before the summary model is called."""

    through_seq: int
    summary_messages: tuple[dict[str, Any], ...]
    replacement_conversation: tuple[dict[str, Any], ...]


def _plain_messages(
    messages: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    try:
        copied = copy.deepcopy(list(messages))
        json.dumps(copied, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError):
        raise ProjectionError(
            "conversation must be safely JSON serializable"
        ) from None
    return copied


def atomic_groups(
    messages: Sequence[Mapping[str, Any]],
) -> tuple[AtomicGroup, ...]:
    """Split a valid conversation without separating calls from results."""

    validate_conversation(messages)
    copied = _plain_messages(messages)
    position = 0
    if copied and copied[0].get("role") == "system":
        position = 1

    groups: list[AtomicGroup] = []
    while position < len(copied):
        message = copied[position]
        role = message["role"]
        if role == "tool":
            raise ProjectionError("tool message cannot begin an atomic group")
        if role != "assistant" or not message.get("tool_calls"):
            groups.append(AtomicGroup(len(groups), (message,)))
            position += 1
            continue

        pending = {call["id"] for call in message["tool_calls"]}
        grouped = [message]
        position += 1
        while pending:
            if position >= len(copied):
                raise ProjectionError("tool call group has missing results")
            result = copied[position]
            if result.get("role") != "tool":
                raise ProjectionError(
                    "tool call group is interrupted before all results"
                )
            pending.remove(result["tool_call_id"])
            grouped.append(result)
            position += 1
        groups.append(AtomicGroup(len(groups), tuple(grouped)))
    return tuple(groups)


def _shorten_tool_output(content: str, limit: int) -> str:
    if len(content) <= limit:
        return content
    if limit <= len(_SHORTENING_MARKER):
        return _SHORTENING_MARKER[:limit]
    remaining = limit - len(_SHORTENING_MARKER)
    head = (remaining + 1) // 2
    tail = remaining // 2
    suffix = content[-tail:] if tail else ""
    return content[:head] + _SHORTENING_MARKER + suffix


def _summary_request_messages(
    conversation: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    transcript = json.dumps(
        list(conversation),
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return (
        {"role": "system", "content": SUMMARY_PROMPT},
        {
            "role": "user",
            "content": "Conversation to summarize (JSON):\n" + transcript,
        },
    )


def prepare_compaction(
    messages: Sequence[Mapping[str, Any]],
    *,
    through_seq: int,
    tail_group_count: int = DEFAULT_TAIL_GROUP_COUNT,
    max_summary_input_chars: int = DEFAULT_MAX_SUMMARY_INPUT_CHARS,
    max_old_tool_chars: int = DEFAULT_MAX_OLD_TOOL_CHARS,
) -> CompactionDraft:
    """Prepare a deterministic summary request and complete replacement tail."""

    if type(through_seq) is not int or through_seq < 1:
        raise ValueError("through_seq must be a positive integer")
    for name, value in (
        ("tail_group_count", tail_group_count),
        ("max_summary_input_chars", max_summary_input_chars),
        ("max_old_tool_chars", max_old_tool_chars),
    ):
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer")

    copied = _plain_messages(messages)
    system_messages = [
        message for message in copied if message.get("role") == "system"
    ]
    groups = atomic_groups(copied)
    user_group_indexes = [
        group.index
        for group in groups
        if group.messages[0].get("role") == "user"
    ]
    if not user_group_indexes:
        raise CompactionError("conversation has no user task to compact")

    tail_start = max(0, len(groups) - tail_group_count)
    retained_indexes = {
        user_group_indexes[0],
        user_group_indexes[-1],
        *range(tail_start, len(groups)),
    }
    replacement = tuple(
        copy.deepcopy(message)
        for group in groups
        if group.index in retained_indexes
        for message in group.messages
    )
    validate_conversation(replacement)

    summary_conversation = [
        *system_messages,
        *(
            message
            for group in groups
            for message in group.messages
        ),
    ]
    serialized = json.dumps(
        summary_conversation,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    if len(serialized) > max_summary_input_chars:
        old_group_indexes = {
            group.index for group in groups if group.index not in retained_indexes
        }
        shortened_any = False
        for group in groups:
            if group.index not in old_group_indexes:
                continue
            for message in group.messages:
                if message.get("role") != "tool":
                    continue
                old_content = message["content"]
                new_content = _shorten_tool_output(
                    old_content, max_old_tool_chars
                )
                if new_content != old_content:
                    message["content"] = new_content
                    shortened_any = True
        serialized = json.dumps(
            summary_conversation,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        if not shortened_any or len(serialized) > max_summary_input_chars:
            raise CompactionError(
                "summary input remains too large after shortening old tool outputs"
            )

    return CompactionDraft(
        through_seq=through_seq,
        summary_messages=_summary_request_messages(summary_conversation),
        replacement_conversation=replacement,
    )


def finalize_checkpoint(
    draft: CompactionDraft, summary: str
) -> dict[str, Any]:
    """Build a checkpoint payload after validating the model handoff."""

    if not isinstance(draft, CompactionDraft):
        raise TypeError("draft must be a CompactionDraft")
    if not isinstance(summary, str) or not summary.strip():
        raise CompactionError("compaction summary must be non-empty text")
    replacement = _plain_messages(draft.replacement_conversation)
    if any(message.get("role") == "system" for message in replacement):
        raise CompactionError("replacement conversation must not contain system")
    validate_conversation(replacement)
    return {
        "through_seq": draft.through_seq,
        "summary": summary,
        "replacement_conversation": replacement,
    }


class SessionCompactor:
    """Create one durable checkpoint at a complete projection boundary."""

    def __init__(
        self,
        *,
        store: RolloutStore,
        state: SessionState,
        model: SummaryModel,
        environment: Callable[
            [], ProjectionEnvironment | Mapping[str, Any]
        ],
        tail_group_count: int = DEFAULT_TAIL_GROUP_COUNT,
        max_summary_input_chars: int = DEFAULT_MAX_SUMMARY_INPUT_CHARS,
        max_old_tool_chars: int = DEFAULT_MAX_OLD_TOOL_CHARS,
    ) -> None:
        if state.session_id != store.session_id:
            raise ValueError("state and store must belong to the same session")
        self.store = store
        self.state = state
        self.model = model
        self.environment = environment
        self.tail_group_count = tail_group_count
        self.max_summary_input_chars = max_summary_input_chars
        self.max_old_tool_chars = max_old_tool_chars

    def __call__(self) -> bool:
        return self.compact()

    def compact(self) -> bool:
        events = self.store.load()
        if self.state.session_id is None or not events:
            raise CompactionError("session has not been created")
        current_environment = self.environment()
        try:
            messages = PromptProjector.project(
                events, self.state, current_environment
            )
            draft = prepare_compaction(
                messages,
                through_seq=self.state.last_seq,
                tail_group_count=self.tail_group_count,
                max_summary_input_chars=self.max_summary_input_chars,
                max_old_tool_chars=self.max_old_tool_chars,
            )
        except (DomainError, ProjectionError) as error:
            raise CompactionError(
                f"compaction requires a completed sampling boundary: {error}"
            ) from error

        try:
            sampled = self.model.sample(
                draft.summary_messages, [], False
            )
        except Exception as error:
            raise CompactionError("summary sampling failed") from error
        if (
            sampled.outcome is not SamplingOutcome.COMPLETE_TEXT
            or not sampled.content.strip()
            or bool(sampled.tool_calls)
        ):
            raise CompactionError(
                "summary model did not return non-empty complete text"
            )

        payload = finalize_checkpoint(draft, sampled.content)
        candidate = Event.create(
            seq=self.state.last_seq + 1,
            session_id=self.store.session_id,
            event_type="compaction_checkpoint",
            payload=payload,
        )
        try:
            candidate_state = reduce_event(self.state, candidate)
            PromptProjector.project(
                (*events, candidate), candidate_state, current_environment
            )
        except (DomainError, ProjectionError) as error:
            raise CompactionError(
                f"checkpoint candidate is invalid: {error}"
            ) from error

        try:
            durable = self.store.append(candidate)
        except Exception as error:
            raise CompactionError("checkpoint append failed") from error
        try:
            SessionReducer.apply(self.state, durable)
        except Exception as error:
            raise CompactionError(
                "durable checkpoint could not be applied"
            ) from error
        return True


__all__ = [
    "AtomicGroup",
    "CompactionDraft",
    "CompactionError",
    "DEFAULT_MAX_OLD_TOOL_CHARS",
    "DEFAULT_MAX_SUMMARY_INPUT_CHARS",
    "DEFAULT_TAIL_GROUP_COUNT",
    "SUMMARY_PROMPT",
    "SUMMARY_SECTIONS",
    "SessionCompactor",
    "atomic_groups",
    "finalize_checkpoint",
    "prepare_compaction",
]
