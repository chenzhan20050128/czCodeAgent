"""Read-only projections that summarize and render a session's facts.

Everything here is a pure function over ``SessionState`` (itself derived from
the append-only rollout by the same ``SessionReducer`` the agent loop uses) or
a read-only directory listing. Nothing acquires a lock, samples a model, or
writes to disk.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from .domain import SessionState, TurnStatus


_MAX_RENDERED_FIELD_CHARS = 500
_TRUNCATION_SUFFIX = " ... [truncated]"


@dataclass(frozen=True)
class SessionSummary:
    """A one-line-friendly digest of a session's current derived state."""

    session_id: str | None
    created_at: str | None
    model: str | None
    turn_count: int
    tool_call_count: int
    last_turn_status: str | None
    is_active: bool
    recovery_blocked: bool

    def render_line(self) -> str:
        session = self.session_id or "<uncreated>"
        status = self.last_turn_status or "-"
        flags = []
        if self.is_active:
            flags.append("active")
        if self.recovery_blocked:
            flags.append("recovery-blocked")
        suffix = f" [{','.join(flags)}]" if flags else ""
        return (
            f"{session}  model={self.model or '-'}  "
            f"turns={self.turn_count}  tools={self.tool_call_count}  "
            f"last={status}{suffix}"
        )


def summarize(state: SessionState) -> SessionSummary:
    """Derive a stable digest from an already-replayed session state."""

    if not isinstance(state, SessionState):
        raise TypeError("state must be a SessionState")
    last_turn_status = _last_turn_status(state)
    return SessionSummary(
        session_id=state.session_id,
        created_at=state.created_at,
        model=state.model,
        turn_count=len(state.turns),
        tool_call_count=len(state.tool_calls),
        last_turn_status=last_turn_status,
        is_active=state.active_turn_id is not None,
        recovery_blocked=state.recovery_blocked,
    )


def render_transcript(state: SessionState) -> str:
    """Render the session's ordered facts as bounded, human-readable text."""

    if not isinstance(state, SessionState):
        raise TypeError("state must be a SessionState")
    if state.session_id is None:
        return "no session has been created yet."

    lines: list[str] = [
        f"session {state.session_id}",
        f"created {state.created_at}  model {state.model}",
        "",
    ]
    for event in state.events:
        rendered = _render_event(event, state)
        if rendered is not None:
            lines.append(rendered)
    summary = summarize(state)
    lines.append("")
    lines.append(summary.render_line())
    return "\n".join(lines)


def list_session_ids(sessions_root: str | os.PathLike[str]) -> list[str]:
    """Return sorted canonical-UUID session ids present under ``sessions_root``."""

    root = Path(sessions_root)
    try:
        entries = list(root.iterdir())
    except (FileNotFoundError, NotADirectoryError):
        return []
    session_ids: list[str] = []
    for entry in entries:
        if entry.suffix != ".jsonl":
            continue
        stem = entry.stem
        if _is_canonical_uuid(stem):
            session_ids.append(stem)
    return sorted(session_ids)


def _render_event(event, state: SessionState) -> str | None:
    payload = event.payload
    if event.type == "turn_started":
        user_input = payload.get("user_input", payload.get("input", ""))
        return f"[user] {_bounded(str(user_input))}"
    if event.type == "assistant_accepted":
        return _render_assistant(payload)
    if event.type == "tool_finished":
        return _render_tool_finished(payload, state)
    if event.type == "tool_reconciled":
        outcome = payload.get("outcome")
        return f"  [reconciled] {outcome}"
    if event.type == "compaction_checkpoint":
        return f"[compacted through seq {payload.get('through_seq')}]"
    if event.type == "undo_finished":
        return f"[undo {payload.get('status')}]"
    if event.type == "turn_finished":
        status = payload.get("status")
        error = payload.get("error")
        suffix = f" ({_bounded(str(error))})" if error else ""
        return f"[turn {status}{suffix}]"
    return None


def _render_assistant(payload) -> str | None:
    content = payload.get("content")
    calls = payload.get("tool_calls", ())
    parts: list[str] = []
    if content:
        parts.append(f"[assistant] {_bounded(str(content))}")
    for call in calls:
        function = call.get("function", {}) if isinstance(call, Mapping) else {}
        name = function.get("name", "?")
        arguments = function.get("arguments", "")
        parts.append(f"  -> {name} {_bounded(str(arguments))}")
    return "\n".join(parts) if parts else None


def _render_tool_finished(payload, state: SessionState) -> str:
    call_key = payload.get("call_key")
    name = "?"
    if isinstance(call_key, str):
        call = state.tool_calls.get(call_key)
        if call is not None:
            name = call.name
    status = payload.get("status", "?")
    result = payload.get("result", "")
    return f"  [{name}: {status}] {_bounded(str(result))}"


def _last_turn_status(state: SessionState) -> str | None:
    for event in reversed(state.events):
        if event.type == "turn_started":
            turn_id = event.payload.get("turn_id")
            status = state.turns.get(turn_id) if isinstance(turn_id, str) else None
            if isinstance(status, TurnStatus):
                return status.value
            return None
    return None


def _bounded(value: str) -> str:
    collapsed = value.replace("\n", " ").replace("\r", " ")
    if len(collapsed) <= _MAX_RENDERED_FIELD_CHARS:
        return collapsed
    keep = _MAX_RENDERED_FIELD_CHARS - len(_TRUNCATION_SUFFIX)
    return collapsed[:keep] + _TRUNCATION_SUFFIX


def _is_canonical_uuid(value: str) -> bool:
    try:
        return str(uuid.UUID(value)) == value
    except (ValueError, AttributeError, TypeError):
        return False


__all__ = [
    "SessionSummary",
    "list_session_ids",
    "render_transcript",
    "summarize",
]
