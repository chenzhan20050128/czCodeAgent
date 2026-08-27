"""Pure validation for MCA's canonical Chat Completions subset."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


class ProjectionError(ValueError):
    """Raised when data cannot form a valid provider conversation."""


# Low-level name for domain consumers.  ProjectionError remains the public
# compatibility name exported by mca.projection.
ConversationError = ProjectionError


def _validate_message_fields(
    message: Mapping[str, Any], allowed: frozenset[str], *, role: str
) -> None:
    fields = set(message)
    if fields != allowed:
        missing = sorted(allowed - fields)
        extra = sorted(fields - allowed)
        raise ConversationError(
            f"{role} message fields mismatch (missing={missing}, extra={extra})"
        )


def _validate_canonical_tool_call(document: object) -> str:
    if not isinstance(document, Mapping):
        raise ConversationError("assistant tool call must be an object")
    _validate_message_fields(
        document, frozenset({"id", "type", "function"}), role="tool call"
    )
    if document["type"] != "function":
        raise ConversationError("only function tool calls are canonical")
    call_id = document["id"]
    if not isinstance(call_id, str) or not call_id:
        raise ConversationError(
            "assistant tool call id must be a non-empty string"
        )

    function = document["function"]
    if not isinstance(function, Mapping):
        raise ConversationError("assistant tool function must be an object")
    _validate_message_fields(
        function, frozenset({"name", "arguments"}), role="tool function"
    )
    if not isinstance(function["name"], str) or not function["name"]:
        raise ConversationError("assistant tool name must be a non-empty string")
    if not isinstance(function["arguments"], str):
        raise ConversationError("canonical tool arguments must be a string")
    return call_id


def validate_conversation(messages: Sequence[Mapping[str, Any]]) -> None:
    """Validate the strict Chat Completions subset emitted by MCA.

    A tool batch is atomic in the model-visible conversation: after an
    assistant requests calls, exactly one result for every ID must appear
    before another assistant or user message.  IDs may be reused by a later
    assistant after the earlier batch is closed.
    """

    if not isinstance(messages, (list, tuple)):
        raise ConversationError("conversation must be an array")

    pending: set[str] = set()
    completed_in_batch: set[str] = set()
    conversation_started = False
    system_seen = False
    for index, message in enumerate(messages):
        if not isinstance(message, Mapping):
            raise ConversationError(f"message {index} must be an object")
        role = message.get("role")
        if role not in {"system", "user", "assistant", "tool"}:
            raise ConversationError(
                f"message {index} has invalid role: {role!r}"
            )

        if pending and role != "tool":
            raise ConversationError(
                f"{role} message appears before every tool call has a result"
            )

        if role == "system":
            _validate_message_fields(
                message, frozenset({"role", "content"}), role=role
            )
            if system_seen or conversation_started:
                raise ConversationError(
                    "conversation may contain only one leading system message"
                )
            if not isinstance(message["content"], str):
                raise ConversationError("system message content must be a string")
            system_seen = True
            continue

        conversation_started = True
        if role == "user":
            _validate_message_fields(
                message, frozenset({"role", "content"}), role=role
            )
            if not isinstance(message["content"], str):
                raise ConversationError("user message content must be a string")
            continue

        if role == "assistant":
            allowed = frozenset({"role", "content"})
            if "tool_calls" in message:
                allowed = allowed | {"tool_calls"}
            if "reasoning_content" in message:
                allowed = allowed | {"reasoning_content"}
            _validate_message_fields(message, allowed, role=role)
            content = message["content"]
            if content is not None and not isinstance(content, str):
                raise ConversationError(
                    "assistant message content must be a string or null"
                )
            if "reasoning_content" in message and not isinstance(
                message["reasoning_content"], str
            ):
                raise ConversationError(
                    "assistant reasoning_content must be a string"
                )
            raw_calls = message.get("tool_calls", ())
            if not isinstance(raw_calls, (list, tuple)):
                raise ConversationError("assistant tool_calls must be an array")
            call_ids = [
                _validate_canonical_tool_call(call) for call in raw_calls
            ]
            if len(call_ids) != len(set(call_ids)):
                raise ConversationError(
                    "assistant message contains a duplicate tool call ID"
                )
            if content is None and not call_ids:
                raise ConversationError(
                    "assistant message must contain text or tool calls"
                )
            pending = set(call_ids)
            completed_in_batch = set()
            continue

        _validate_message_fields(
            message,
            frozenset({"role", "tool_call_id", "content"}),
            role=role,
        )
        call_id = message["tool_call_id"]
        if not isinstance(call_id, str) or not call_id:
            raise ConversationError("tool_call_id must be a non-empty string")
        if not isinstance(message["content"], str):
            raise ConversationError("tool message content must be a string")
        if call_id in completed_in_batch:
            raise ConversationError(
                f"duplicate tool result for call ID {call_id!r}"
            )
        if call_id not in pending:
            raise ConversationError(f"orphan tool result for call ID {call_id!r}")
        pending.remove(call_id)
        completed_in_batch.add(call_id)

    if pending:
        missing = sorted(pending)
        raise ConversationError(f"tool calls are missing a result: {missing}")


__all__ = ["ConversationError", "ProjectionError", "validate_conversation"]
