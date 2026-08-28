"""Pure session resume and recovery-reconciliation orchestration."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from .domain import (
    DomainError,
    Event,
    SessionReducer,
    SessionState,
    ToolStatus,
    TurnStatus,
    plan_recovery_events,
    reduce_event,
)
from .store import RolloutStore


class ResumeError(RuntimeError):
    """Raised when a session cannot be resumed safely."""


class ReconciliationError(RuntimeError):
    """Raised when an unknown tool outcome cannot be reconciled safely."""


@dataclass
class ResumedSession:
    """A replayed session whose store lock remains held until close."""

    store: RolloutStore
    state: SessionState
    workspace: Path

    def close(self) -> None:
        self.store.close()

    def __enter__(self) -> ResumedSession:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


def _canonical_workspace(
    workspace: str | os.PathLike[str], *, label: str
) -> Path:
    raw_workspace = os.fspath(workspace)
    try:
        candidate = Path(raw_workspace).resolve(strict=True)
    except (FileNotFoundError, OSError) as error:
        raise ResumeError(f"{label} workspace does not exist") from error
    if not candidate.is_dir():
        raise ResumeError(f"{label} workspace must be a directory")
    if label == "requested" and (
        not Path(raw_workspace).is_absolute() or str(candidate) != raw_workspace
    ):
        raise ResumeError(
            f"{label} workspace must be a canonical absolute path"
        )
    return candidate


def _prevalidate(events: list[Event], state: SessionState) -> None:
    working = state
    for event in events:
        working = reduce_event(working, event)


def _append_candidates(
    store: RolloutStore,
    state: SessionState,
    candidates: list[Event],
    *,
    error_type: type[ResumeError] | type[ReconciliationError],
) -> list[Event]:
    try:
        _prevalidate(candidates, state)
    except DomainError as error:
        raise error_type(f"candidate recovery facts are invalid: {error}") from error

    durable_events: list[Event] = []
    for candidate in candidates:
        try:
            durable = store.append(candidate)
        except BaseException as error:
            store.close()
            if not isinstance(error, Exception):
                raise
            raise error_type("recovery event append failed") from error
        try:
            SessionReducer.apply(state, durable)
        except BaseException as error:
            store.close()
            if not isinstance(error, Exception):
                raise
            raise error_type(
                "durable recovery event could not be applied"
            ) from error
        durable_events.append(durable)
    return durable_events


def _finish_candidate(
    state: SessionState, *, status: TurnStatus, error: str
) -> Event:
    turn_id = state.active_turn_id
    if turn_id is None or state.session_id is None:
        raise DomainError("cannot finish a session without an active turn")
    return Event.create(
        seq=state.last_seq + 1,
        session_id=state.session_id,
        event_type="turn_finished",
        payload={
            "turn_id": turn_id,
            "status": status.value,
            "error": error,
        },
    )


def _intent_candidate(
    state: SessionState, *, action: str, reason: str
) -> Event:
    turn_id = state.active_turn_id
    if turn_id is None or state.session_id is None:
        raise DomainError("cannot recover a session without an active turn")
    return Event.create(
        seq=state.last_seq + 1,
        session_id=state.session_id,
        event_type="turn_recovery_intent",
        payload={
            "turn_id": turn_id,
            "action": action,
            "reason": reason,
        },
    )


def _abandon_candidates(state: SessionState, *, note: str) -> list[Event]:
    if state.session_id is None:
        raise DomainError("cannot reconcile before session creation")
    candidates: list[Event] = []
    next_seq = state.last_seq + 1
    for call in state.tool_calls.values():
        if (
            call.turn_id == state.active_turn_id
            and call.status is ToolStatus.OUTCOME_UNKNOWN
        ):
            candidates.append(
                Event.create(
                    seq=next_seq,
                    session_id=state.session_id,
                    event_type="tool_reconciled",
                    payload={
                        "call_key": call.call_key,
                        "call_id": call.provider_call_id,
                        "outcome": "abandoned",
                        "note": note,
                    },
                )
            )
            next_seq += 1
    return candidates


def _complete_recovery_intent(
    store: RolloutStore,
    state: SessionState,
    *,
    error_type: type[ResumeError] | type[ReconciliationError],
) -> list[Event]:
    intent = state.pending_recovery_intent
    if intent is None:
        raise DomainError("no recovery intent is pending")
    action = intent.payload["action"]
    reason = intent.payload["reason"]
    if action == "recover_interrupted":
        candidates = plan_recovery_events(state)
        finish_status = TurnStatus.INTERRUPTED
    else:
        candidates = _abandon_candidates(state, note=reason)
        finish_status = TurnStatus.ABANDONED

    durable: list[Event] = []
    if candidates:
        durable.extend(
            _append_candidates(
                store, state, candidates, error_type=error_type
            )
        )
    if state.recovery_blocked:
        return durable
    finish = _finish_candidate(
        state, status=finish_status, error=reason
    )
    durable.extend(
        _append_candidates(store, state, [finish], error_type=error_type)
    )
    return durable


def _recover_open_calls(store: RolloutStore, state: SessionState) -> None:
    if state.pending_recovery_intent is not None:
        _complete_recovery_intent(
            store, state, error_type=ResumeError
        )
        return

    has_requested = any(
        call.status is ToolStatus.REQUESTED for call in state.tool_calls.values()
    )
    has_started = any(
        call.status is ToolStatus.STARTED for call in state.tool_calls.values()
    )
    has_unknown = any(
        call.status is ToolStatus.OUTCOME_UNKNOWN
        for call in state.tool_calls.values()
    )
    if (
        has_requested
        and not has_started
        and not has_unknown
        and not state.recovery_blocked
    ):
        intent = _intent_candidate(
            state,
            action="recover_interrupted",
            reason="turn interrupted before requested tools were started",
        )
        _append_candidates(store, state, [intent], error_type=ResumeError)
        _complete_recovery_intent(
            store, state, error_type=ResumeError
        )
        return

    candidates = plan_recovery_events(state)
    if candidates:
        _append_candidates(store, state, candidates, error_type=ResumeError)


def resume_session(
    sessions_root: str | os.PathLike[str],
    session_id: str,
    workspace: str | os.PathLike[str],
) -> ResumedSession:
    """Open, lock, replay, validate, and recover one local session."""

    requested_workspace = _canonical_workspace(workspace, label="requested")
    try:
        store = RolloutStore.open(sessions_root, session_id)
    except FileNotFoundError as error:
        raise ResumeError(f"session {session_id!r} does not exist") from error

    try:
        events = store.load()
        if not events:
            raise ResumeError("session rollout is empty")
        state = SessionReducer.replay(events)
        if state.cwd is None:
            raise ResumeError("session has no recorded cwd")
        recorded_workspace = _canonical_workspace(state.cwd, label="recorded")
        if str(recorded_workspace) != state.cwd:
            raise ResumeError(
                "recorded cwd must be a canonical absolute path"
            )
        if recorded_workspace != requested_workspace:
            raise ResumeError("requested cwd does not match session cwd")
        _recover_open_calls(store, state)
        return ResumedSession(store, state, requested_workspace)
    except BaseException:
        store.close()
        raise


def continuable_turn_id(state: SessionState) -> str | None:
    """Return the old active Turn that may resume sampling, if any."""

    if not isinstance(state, SessionState):
        raise TypeError("state must be a SessionState")
    if state.pending_recovery_intent is not None:
        raise ReconciliationError(
            "session has a pending recovery intent"
        )
    if state.recovery_blocked:
        raise ReconciliationError(
            "session recovery is blocked by an unknown tool outcome"
        )
    turn_id = state.active_turn_id
    if turn_id is None:
        return None
    if state.turns.get(turn_id) is not TurnStatus.ACTIVE:
        raise ReconciliationError("active turn has an inconsistent status")
    return turn_id


def reconcile_tool(
    store: RolloutStore,
    state: SessionState,
    call_key: str,
    outcome: str,
    note: str = "",
) -> Event:
    """Durably reconcile one unknown call, or abandon its entire Turn."""

    if state.session_id != store.session_id:
        raise ReconciliationError(
            "state and store must belong to the same session"
        )
    if not isinstance(call_key, str) or not call_key:
        raise ReconciliationError("call_key must be a non-empty string")
    if outcome not in {"succeeded", "failed", "abandoned"}:
        raise ReconciliationError(
            "outcome must be succeeded, failed, or abandoned"
        )
    if not isinstance(note, str):
        raise TypeError("note must be a string")
    target = state.tool_calls.get(call_key)
    if target is None:
        raise ReconciliationError(f"unknown tool call key: {call_key}")
    if target.status is not ToolStatus.OUTCOME_UNKNOWN:
        raise ReconciliationError(
            "only an outcome_unknown tool call can be reconciled"
        )
    if state.active_turn_id != target.turn_id:
        raise ReconciliationError("unknown tool call is not in the active turn")
    assert state.session_id is not None

    if state.pending_recovery_intent is not None:
        raise ReconciliationError("a recovery intent is already pending")

    if outcome == "abandoned":
        intent = _intent_candidate(
            state, action="abandon", reason=note
        )
        _append_candidates(
            store, state, [intent], error_type=ReconciliationError
        )
        durable = _complete_recovery_intent(
            store, state, error_type=ReconciliationError
        )
        if not durable:
            raise ReconciliationError(
                "abandon recovery produced no reconciliation facts"
            )
        return durable[0]

    candidates: list[Event] = []
    candidates.append(
        Event.create(
            seq=state.last_seq + 1,
            session_id=state.session_id,
            event_type="tool_reconciled",
            payload={
                "call_key": target.call_key,
                "call_id": target.provider_call_id,
                "outcome": outcome,
                "note": note,
            },
        )
    )

    durable = _append_candidates(
        store, state, candidates, error_type=ReconciliationError
    )
    return durable[0]


__all__ = [
    "ReconciliationError",
    "ResumedSession",
    "ResumeError",
    "continuable_turn_id",
    "reconcile_tool",
    "resume_session",
]
