"""Domain values and validation for the mca event log."""

from __future__ import annotations

import json
import math
import uuid
import base64
import binascii
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from enum import Enum
from types import MappingProxyType
from typing import Any, Iterable

from .conversation import ConversationError, validate_conversation


EVENT_VERSION = 1
_EVENT_FIELDS = {
    "version",
    "seq",
    "event_id",
    "timestamp",
    "session_id",
    "type",
    "payload",
}
EVENT_FIELDS = frozenset(_EVENT_FIELDS)


class DomainError(ValueError):
    """Raised when persisted data violates the domain contract."""


class SamplingOutcome(str, Enum):
    """Classification of a complete model sampling attempt."""

    COMPLETE_TEXT = "complete_text"
    VALID_TOOL_BATCH = "valid_tool_batch"
    LENGTH_EXCEEDED = "length_exceeded"
    CONTEXT_OVERFLOW = "context_overflow"
    FILTERED = "filtered"
    TRANSPORT_INTERRUPTED = "transport_interrupted"
    PROTOCOL_ERROR = "protocol_error"
    ABORTED = "aborted"


class ToolStatus(str, Enum):
    """Lifecycle state of one accepted tool call."""

    REQUESTED = "requested"
    STARTED = "started"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    DENIED = "denied"
    INVALID_ARGUMENTS = "invalid_arguments"
    UNKNOWN_TOOL = "unknown_tool"
    TIMED_OUT = "timed_out"
    INTERRUPTED = "interrupted"
    CANCELLED = "cancelled"
    NOT_EXECUTED = "not_executed"
    BATCH_LIMIT_EXCEEDED = "batch_limit_exceeded"
    CONFLICT = "conflict"
    OUTCOME_UNKNOWN = "outcome_unknown"
    USER_CONFIRMED_SUCCESS = "user_confirmed_success"
    USER_CONFIRMED_FAILURE = "user_confirmed_failure"
    ABANDONED = "abandoned"


class TurnStatus(str, Enum):
    """Lifecycle state of a user turn."""

    ACTIVE = "active"
    RECOVERY_BLOCKED = "recovery_blocked"
    COMPLETED = "completed"
    FAILED = "failed"
    INTERRUPTED = "interrupted"
    MAX_STEPS_REACHED = "max_steps_reached"
    ABANDONED = "abandoned"


_TERMINAL_TOOL_STATUSES = frozenset(
    status
    for status in ToolStatus
    if status not in {ToolStatus.REQUESTED, ToolStatus.STARTED}
)
_TERMINAL_TURN_STATUSES = frozenset(
    status
    for status in TurnStatus
    if status not in {TurnStatus.ACTIVE, TurnStatus.RECOVERY_BLOCKED}
)
_MANAGED_WRITE_TOOL_NAMES = frozenset({"write_file", "edit_file"})
_UNDO_SUCCESS_STATUSES = frozenset(
    {"restored", "deleted", "already_restored", "already_deleted"}
)
_UNDO_PREFLIGHT_ABORT_STATUSES = frozenset(
    {"conflict", "ineligible", "not_modified"}
)
_RECOVERY_INTENT_ACTIONS = frozenset({"recover_interrupted", "abandon"})
_TOOL_FINISH_TRANSITIONS = {
    ToolStatus.REQUESTED: frozenset(
        {
            ToolStatus.DENIED,
            ToolStatus.INVALID_ARGUMENTS,
            ToolStatus.UNKNOWN_TOOL,
            ToolStatus.NOT_EXECUTED,
            ToolStatus.BATCH_LIMIT_EXCEEDED,
            ToolStatus.CANCELLED,
            ToolStatus.FAILED,
        }
    ),
    ToolStatus.STARTED: frozenset(
        {
            ToolStatus.SUCCEEDED,
            ToolStatus.FAILED,
            ToolStatus.TIMED_OUT,
            ToolStatus.INTERRUPTED,
            ToolStatus.CONFLICT,
            ToolStatus.CANCELLED,
            ToolStatus.OUTCOME_UNKNOWN,
        }
    ),
}


def _canonical_uuid(value: object, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise DomainError(f"{field_name} must be a UUID string")
    try:
        parsed = uuid.UUID(value)
    except (ValueError, AttributeError, TypeError):
        raise DomainError(f"{field_name} must be a canonical UUID") from None
    if str(parsed) != value:
        raise DomainError(f"{field_name} must be a canonical UUID")
    return value


def _validate_timestamp(value: object) -> str:
    if not isinstance(value, str):
        raise DomainError("timestamp must be an ISO UTC string")
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        raise DomainError("timestamp must be an ISO UTC string") from None
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(None):
        raise DomainError("timestamp must be an ISO UTC string")
    return value


def _freeze_json(value: Any, *, path: str = "payload") -> Any:
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise DomainError(f"{path} contains a non-finite number")
        return value
    if isinstance(value, list) or isinstance(value, tuple):
        return tuple(
            _freeze_json(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        )
    if isinstance(value, Mapping):
        frozen: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise DomainError(f"{path} keys must be strings")
            frozen[key] = _freeze_json(item, path=f"{path}.{key}")
        return MappingProxyType(frozen)
    raise DomainError(f"{path} contains a non-JSON value")


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


@dataclass(frozen=True)
class Event:
    """One immutable, validated fact in a session rollout."""

    version: int
    seq: int
    event_id: str
    timestamp: str
    session_id: str
    type: str
    payload: Mapping[str, Any]

    def __post_init__(self) -> None:
        if type(self.version) is not int or self.version != EVENT_VERSION:
            raise DomainError(f"unsupported event version: {self.version!r}")
        if type(self.seq) is not int or self.seq < 1:
            raise DomainError("seq must be a positive integer")
        _canonical_uuid(self.event_id, field_name="event_id")
        _validate_timestamp(self.timestamp)
        _canonical_uuid(self.session_id, field_name="session_id")
        if not isinstance(self.type, str) or not self.type:
            raise DomainError("type must be a non-empty string")
        if not isinstance(self.payload, Mapping):
            raise DomainError("payload must be an object")
        object.__setattr__(self, "payload", _freeze_json(self.payload))

    @classmethod
    def create(
        cls,
        *,
        seq: int,
        session_id: str,
        event_type: str,
        payload: Mapping[str, Any],
    ) -> Event:
        timestamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        return cls(
            version=EVENT_VERSION,
            seq=seq,
            event_id=str(uuid.uuid4()),
            timestamp=timestamp,
            session_id=session_id,
            type=event_type,
            payload=payload,
        )

    @classmethod
    def from_dict(cls, document: object) -> Event:
        if not isinstance(document, dict):
            raise DomainError("event must be a JSON object")
        fields = set(document)
        if fields != _EVENT_FIELDS:
            missing = sorted(_EVENT_FIELDS - fields)
            extra = sorted(fields - _EVENT_FIELDS)
            raise DomainError(
                f"event fields mismatch (missing={missing}, extra={extra})"
            )
        return cls(
            version=document["version"],
            seq=document["seq"],
            event_id=document["event_id"],
            timestamp=document["timestamp"],
            session_id=document["session_id"],
            type=document["type"],
            payload=document["payload"],
        )

    def to_dict(self) -> dict[str, Any]:
        document = {
            "version": self.version,
            "seq": self.seq,
            "event_id": self.event_id,
            "timestamp": self.timestamp,
            "session_id": self.session_id,
            "type": self.type,
            "payload": _thaw_json(self.payload),
        }
        try:
            json.dumps(document, allow_nan=False)
        except (TypeError, ValueError):
            raise DomainError("event is not safely JSON serializable") from None
        return document


@dataclass(frozen=True)
class ToolCall:
    """Immutable derived view of a tool call's current lifecycle state."""

    call_key: str
    provider_call_id: str
    turn_id: str
    name: str
    arguments: str
    status: ToolStatus = ToolStatus.REQUESTED
    approved: bool | None = None
    approval_scope: str | None = None
    result: str | None = None
    exit_code: int | None = None
    truncated: bool = False
    recovery_blocked: bool = False
    reconciliation_note: str | None = None
    requested_seq: int | None = None
    started_seq: int | None = None
    finished_seq: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.call_key, str) or not self.call_key:
            raise DomainError("call_key must be a non-empty string")
        if not isinstance(self.provider_call_id, str) or not self.provider_call_id:
            raise DomainError("provider_call_id must be a non-empty string")
        _canonical_uuid(self.turn_id, field_name="turn_id")
        if not isinstance(self.name, str) or not self.name:
            raise DomainError("tool name must be a non-empty string")
        if not isinstance(self.arguments, str):
            raise DomainError("tool arguments must be a JSON string")
        if not isinstance(self.status, ToolStatus):
            raise DomainError("status must be a ToolStatus")

    @property
    def call_id(self) -> str:
        """Compatibility alias for the provider-visible call ID."""

        return self.provider_call_id

    @property
    def is_terminal(self) -> bool:
        return self.status in _TERMINAL_TOOL_STATUSES


@dataclass(frozen=True)
class FileSnapshot:
    """The first pre-write baseline for a path in one turn."""

    turn_id: str
    path: str
    existed_before: bool
    before_bytes: str
    before_mode: int | None
    after_hash: str | None = None
    source_call_key: str | None = None
    after_mode: int | None = None
    created_directories: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _canonical_uuid(self.turn_id, field_name="turn_id")
        if not isinstance(self.path, str) or not self.path:
            raise DomainError("snapshot path must be a non-empty string")
        if type(self.existed_before) is not bool:
            raise DomainError("existed_before must be a boolean")
        if not isinstance(self.before_bytes, str):
            raise DomainError("before_bytes must be a string")
        try:
            decoded_before = base64.b64decode(
                self.before_bytes.encode("ascii"), validate=True
            )
        except (UnicodeEncodeError, binascii.Error):
            raise DomainError("before_bytes must be valid base64") from None
        if self.existed_before and self.before_mode is None:
            raise DomainError("before_mode is required for an existing file")
        if not self.existed_before and self.before_mode is not None:
            raise DomainError("before_mode must be null for a new file")
        if not self.existed_before and decoded_before:
            raise DomainError("a new file must have an empty baseline")
        if self.before_mode is not None and (
            type(self.before_mode) is not int or not 0 <= self.before_mode <= 0o7777
        ):
            raise DomainError("before_mode must be permission bits or null")
        if self.after_hash is not None and (not isinstance(self.after_hash, str) or not self.after_hash):
            raise DomainError("after_hash must be a non-empty string or null")
        if self.source_call_key is not None and (
            not isinstance(self.source_call_key, str) or not self.source_call_key
        ):
            raise DomainError("source_call_key must be a non-empty string or null")
        if self.after_mode is not None and (
            type(self.after_mode) is not int or not 0 <= self.after_mode <= 0o7777
        ):
            raise DomainError("after_mode must be permission bits or null")
        if self.after_hash is None and self.after_mode is not None:
            raise DomainError("after_mode requires after_hash")
        if not isinstance(self.created_directories, tuple) or any(
            not isinstance(item, str) or not item
            for item in self.created_directories
        ):
            raise DomainError("created_directories must be non-empty strings")
        if self.created_directories and self.after_hash is None:
            raise DomainError("created_directories requires a successful write")
        if self.created_directories and self.existed_before:
            raise DomainError("created_directories requires a new file")


@dataclass
class SessionState:
    """Mutable state derived exclusively by replaying immutable events."""

    session_id: str | None = None
    cwd: str | None = None
    model: str | None = None
    context_window: int | None = None
    created_at: str | None = None
    last_seq: int = 0
    last_event_timestamp: str | None = None
    active_turn_id: str | None = None
    turns: dict[str, TurnStatus] = field(default_factory=dict)
    turn_inputs: dict[str, str] = field(default_factory=dict)
    tool_calls: dict[str, ToolCall] = field(default_factory=dict)
    file_snapshots: dict[tuple[str, str], FileSnapshot] = field(
        default_factory=dict
    )
    undo_results: dict[str, Event] = field(default_factory=dict)
    latest_checkpoint: Event | None = None
    pending_recovery_intent: Event | None = None
    assistant_events: list[Event] = field(default_factory=list)
    events: list[Event] = field(default_factory=list)
    recovery_blocked: bool = False
    last_usage: tuple[int, int, int] | None = None
    plan_mode_active: bool = False


def _payload_string(
    payload: Mapping[str, Any], key: str, *, allow_empty: bool = False
) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or (not allow_empty and not value):
        qualifier = "a string" if allow_empty else "a non-empty string"
        raise DomainError(f"{key} must be {qualifier}")
    return value


def _payload_uuid(payload: Mapping[str, Any], key: str) -> str:
    return _canonical_uuid(_payload_string(payload, key), field_name=key)


def _tool_status(value: object) -> ToolStatus:
    if not isinstance(value, str):
        raise DomainError("tool status must be a string")
    aliases = {"success": "succeeded", "timeout": "timed_out"}
    try:
        return ToolStatus(aliases.get(value, value))
    except ValueError:
        raise DomainError(f"unknown tool status: {value!r}") from None


def _turn_status(value: object) -> TurnStatus:
    if not isinstance(value, str):
        raise DomainError("turn status must be a string")
    try:
        status = TurnStatus(value)
    except ValueError:
        raise DomainError(f"unknown turn status: {value!r}") from None
    if status not in _TERMINAL_TURN_STATUSES:
        raise DomainError(f"turn_finished cannot use status {value!r}")
    return status


def _normalized_created_directories(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, (list, tuple)) or any(
        not isinstance(item, str) or not item for item in value
    ):
        raise DomainError("created_directories must be an array of non-empty strings")
    return tuple(value)


def _parse_usage_payload(value: object) -> tuple[int, int, int] | None:
    """Validate optional provider token usage recorded on an assistant fact."""

    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise DomainError("usage must be an object or null")
    fields = set(value)
    expected = {"prompt_tokens", "completion_tokens", "total_tokens"}
    if fields != expected:
        raise DomainError(
            "usage must have exactly prompt_tokens, completion_tokens, total_tokens"
        )
    parsed: list[int] = []
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        token_count = value[key]
        if type(token_count) is not int or token_count < 0:
            raise DomainError(f"usage {key} must be a non-negative integer")
        parsed.append(token_count)
    return (parsed[0], parsed[1], parsed[2])


def reduce_undo_status(statuses: Iterable[str]) -> str:
    """Derive the durable undo outcome from per-file outcomes."""

    observed = set(statuses)
    if observed <= _UNDO_SUCCESS_STATUSES:
        return "succeeded"
    if (
        observed <= _UNDO_PREFLIGHT_ABORT_STATUSES
        and bool(observed & {"conflict", "ineligible"})
    ):
        return "conflict"
    return "partial"


class SessionReducer:
    """Validate and apply rollout facts to a SessionState."""

    @classmethod
    def replay(cls, events: Iterable[Event]) -> SessionState:
        state = SessionState()
        for event in events:
            cls.apply(state, event)
        return state

    @classmethod
    def apply(cls, state: SessionState, event: Event) -> SessionState:
        if not isinstance(state, SessionState) or not isinstance(event, Event):
            raise TypeError("apply requires a SessionState and Event")
        expected_seq = state.last_seq + 1
        if event.seq != expected_seq:
            raise DomainError(
                f"invalid event sequence: expected {expected_seq}, got {event.seq}"
            )
        if state.session_id is not None and event.session_id != state.session_id:
            raise DomainError("event belongs to another session")
        if state.session_id is None and event.type != "session_created":
            raise DomainError("session_created must be the first event")

        handler = getattr(cls, f"_apply_{event.type}", None)
        if handler is None:
            raise DomainError(f"unknown event type: {event.type}")
        handler(state, event)
        state.last_seq = event.seq
        state.last_event_timestamp = event.timestamp
        state.events.append(event)
        return state

    @staticmethod
    def _apply_session_created(state: SessionState, event: Event) -> None:
        if state.session_id is not None:
            raise DomainError("session has already been created")
        cwd = _payload_string(event.payload, "cwd")
        model = _payload_string(event.payload, "model")
        context_window = event.payload.get("context_window")
        if type(context_window) is not int or context_window <= 0:
            raise DomainError("context_window must be a positive integer")
        state.session_id = event.session_id
        state.cwd = cwd
        state.model = model
        state.context_window = context_window
        state.created_at = event.timestamp

    @staticmethod
    def _apply_turn_started(state: SessionState, event: Event) -> None:
        if state.active_turn_id is not None:
            raise DomainError("cannot start a turn while another turn is active")
        turn_id = _payload_uuid(event.payload, "turn_id")
        if turn_id in state.turns:
            raise DomainError("turn_id has already been used")
        user_input = event.payload.get("user_input", event.payload.get("input", ""))
        if not isinstance(user_input, str):
            raise DomainError("user_input must be a string")
        state.active_turn_id = turn_id
        state.turns[turn_id] = TurnStatus.ACTIVE
        state.turn_inputs[turn_id] = user_input

    @staticmethod
    def _apply_assistant_accepted(state: SessionState, event: Event) -> None:
        turn_id = SessionReducer._require_active_turn(state)
        SessionReducer._reject_pending_recovery_action(
            state, "accept an assistant response"
        )
        if state.recovery_blocked:
            raise DomainError(
                "cannot accept an assistant response while recovery is blocked"
            )
        unresolved = any(
            call.turn_id == turn_id
            and call.status in {ToolStatus.REQUESTED, ToolStatus.STARTED}
            for call in state.tool_calls.values()
        )
        if unresolved:
            raise DomainError(
                "cannot accept an assistant response with unresolved tool calls"
            )
        explicit_turn_id = event.payload.get("turn_id")
        if explicit_turn_id is not None and explicit_turn_id != turn_id:
            raise DomainError("assistant event belongs to another turn")
        tool_documents = event.payload.get("tool_calls", [])
        if not isinstance(tool_documents, (list, tuple)):
            raise DomainError("tool_calls must be an array")

        pending: list[ToolCall] = []
        seen: set[str] = set()
        for document in tool_documents:
            if not isinstance(document, Mapping):
                raise DomainError("each tool call must be an object")
            call_id = document.get("id", document.get("call_id"))
            if not isinstance(call_id, str) or not call_id:
                raise DomainError("tool call id must be a non-empty string")
            if call_id in seen:
                raise DomainError(f"duplicate tool call id: {call_id}")
            seen.add(call_id)

            if "function" in document:
                if document.get("type", "function") != "function":
                    raise DomainError("only function tool calls are supported")
                function = document["function"]
                if not isinstance(function, Mapping):
                    raise DomainError("tool call function must be an object")
                name = function.get("name")
                arguments = function.get("arguments")
            else:
                name = document.get("name")
                arguments = document.get("arguments")
            if not isinstance(name, str) or not name:
                raise DomainError("tool call name must be a non-empty string")
            if not isinstance(arguments, str):
                try:
                    arguments = json.dumps(
                        arguments,
                        ensure_ascii=False,
                        allow_nan=False,
                        separators=(",", ":"),
                        sort_keys=True,
                    )
                except (TypeError, ValueError):
                    raise DomainError("tool arguments must be JSON-compatible") from None
            pending.append(
                ToolCall(
                    call_key=f"{event.seq}:{call_id}",
                    provider_call_id=call_id,
                    turn_id=turn_id,
                    name=name,
                    arguments=arguments,
                    requested_seq=event.seq,
                )
            )

        content = event.payload.get("content")
        if content is not None and not isinstance(content, str):
            raise DomainError("assistant content must be a string or null")
        if "reasoning_content" in event.payload and not isinstance(
            event.payload["reasoning_content"], str
        ):
            raise DomainError(
                "assistant reasoning_content must be a string"
            )
        usage = _parse_usage_payload(event.payload.get("usage"))
        for call in pending:
            if call.call_key in state.tool_calls:
                raise DomainError(f"duplicate internal tool call key: {call.call_key}")
            state.tool_calls[call.call_key] = call
        state.assistant_events.append(event)
        if usage is not None:
            state.last_usage = usage

    @staticmethod
    def _apply_approval_decided(state: SessionState, event: Event) -> None:
        SessionReducer._reject_pending_recovery_action(
            state, "record an approval decision"
        )
        call = SessionReducer._call_for_active_turn(state, event.payload)
        if call.is_terminal or call.status is ToolStatus.STARTED:
            raise DomainError("approval decision is too late for this call")
        if call.approved is not None:
            raise DomainError("approval has already been decided")
        decision = event.payload.get("approved", event.payload.get("allow"))
        if decision is None and "decision" in event.payload:
            raw_decision = event.payload["decision"]
            if raw_decision not in {"allow", "deny"}:
                raise DomainError("decision must be allow or deny")
            decision = raw_decision == "allow"
        if type(decision) is not bool:
            raise DomainError("approved must be a boolean")
        scope = event.payload.get("scope")
        if scope is not None and (not isinstance(scope, str) or not scope):
            raise DomainError("scope must be a non-empty string or null")
        state.tool_calls[call.call_key] = replace(
            call, approved=decision, approval_scope=scope
        )

    @staticmethod
    def _apply_tool_started(state: SessionState, event: Event) -> None:
        SessionReducer._reject_pending_recovery_action(
            state, "start a tool call"
        )
        call = SessionReducer._call_for_active_turn(state, event.payload)
        if call.status is not ToolStatus.REQUESTED:
            raise DomainError("only a requested tool call can start")
        if call.approved is False:
            raise DomainError("a denied tool call cannot start")
        state.tool_calls[call.call_key] = replace(
            call, status=ToolStatus.STARTED, started_seq=event.seq
        )

    @staticmethod
    def _apply_tool_finished(state: SessionState, event: Event) -> None:
        call = SessionReducer._call_for_active_turn(state, event.payload)
        if call.is_terminal:
            raise DomainError("tool call already has a terminal result")
        status = _tool_status(event.payload.get("status"))
        intent = state.pending_recovery_intent
        if intent is not None and (
            intent.payload["action"] != "recover_interrupted"
            or status is not ToolStatus.NOT_EXECUTED
        ):
            raise DomainError(
                "tool result does not match the pending recovery intent"
            )
        if status not in _TERMINAL_TOOL_STATUSES:
            raise DomainError("tool_finished requires a terminal status")
        allowed = _TOOL_FINISH_TRANSITIONS.get(call.status, frozenset())
        if status not in allowed:
            raise DomainError(
                f"{call.status.value} tool call cannot finish as {status.value}"
            )

        result = event.payload.get("result")
        if result is not None and not isinstance(result, str):
            raise DomainError("tool result must be a string or null")
        exit_code = event.payload.get("exit_code")
        if exit_code is not None and type(exit_code) is not int:
            raise DomainError("exit_code must be an integer or null")
        truncated = event.payload.get("truncated", False)
        if type(truncated) is not bool:
            raise DomainError("truncated must be a boolean")
        recovery_blocked = event.payload.get("recovery_blocked", False)
        if type(recovery_blocked) is not bool:
            raise DomainError("recovery_blocked must be a boolean")
        if recovery_blocked is not (status is ToolStatus.OUTCOME_UNKNOWN):
            raise DomainError(
                "recovery_blocked must be true exactly for outcome_unknown"
            )

        path = event.payload.get("path")
        after_hash = event.payload.get("after_hash")
        after_mode = event.payload.get("after_mode")
        created_directories = event.payload.get("created_directories")
        snapshot_update: tuple[tuple[str, str], FileSnapshot] | None = None
        if (
            path is not None
            or after_hash is not None
            or after_mode is not None
            or created_directories is not None
        ):
            if status is not ToolStatus.SUCCEEDED:
                raise DomainError("after_hash is only valid for a successful tool")
            if call.name not in _MANAGED_WRITE_TOOL_NAMES:
                raise DomainError("file result metadata requires a managed write call")
            if not isinstance(path, str) or not path:
                raise DomainError("path must accompany after_hash")
            if not isinstance(after_hash, str) or not after_hash:
                raise DomainError("after_hash must accompany path")
            if after_mode is not None and (
                type(after_mode) is not int or not 0 <= after_mode <= 0o7777
            ):
                raise DomainError("after_mode must be permission bits or null")
            directories = _normalized_created_directories(created_directories)
            key = (call.turn_id, path)
            snapshot = state.file_snapshots.get(key)
            if snapshot is None:
                raise DomainError("successful write has no file baseline")
            if (
                snapshot.after_hash is None
                and snapshot.source_call_key != call.call_key
            ):
                raise DomainError("successful write does not match snapshot source call")
            snapshot_update = (
                key,
                replace(
                    snapshot,
                    after_hash=after_hash,
                    after_mode=after_mode,
                    created_directories=directories,
                ),
            )

        state.tool_calls[call.call_key] = replace(
            call,
            status=status,
            result=result,
            exit_code=exit_code,
            truncated=truncated,
            recovery_blocked=recovery_blocked,
            finished_seq=event.seq,
        )
        if snapshot_update is not None:
            state.file_snapshots[snapshot_update[0]] = snapshot_update[1]
        SessionReducer._refresh_recovery_block(state)

    @staticmethod
    def _apply_turn_recovery_intent(
        state: SessionState, event: Event
    ) -> None:
        expected_fields = {"turn_id", "action", "reason"}
        fields = set(event.payload)
        if fields != expected_fields:
            missing = sorted(expected_fields - fields)
            extra = sorted(fields - expected_fields)
            raise DomainError(
                "turn_recovery_intent payload fields mismatch "
                f"(missing={missing}, extra={extra})"
            )
        turn_id = SessionReducer._require_active_turn(state)
        event_turn_id = _payload_uuid(event.payload, "turn_id")
        if event_turn_id != turn_id:
            raise DomainError(
                "turn_recovery_intent belongs to another turn"
            )
        action = _payload_string(event.payload, "action")
        if action not in _RECOVERY_INTENT_ACTIONS:
            raise DomainError(f"unknown recovery intent action: {action!r}")
        reason = event.payload.get("reason")
        if not isinstance(reason, str):
            raise DomainError("recovery intent reason must be a string")
        if state.pending_recovery_intent is not None:
            raise DomainError("a recovery intent is already pending")

        turn_calls = [
            call
            for call in state.tool_calls.values()
            if call.turn_id == turn_id
        ]
        if action == "recover_interrupted":
            if not any(
                call.status is ToolStatus.REQUESTED for call in turn_calls
            ):
                raise DomainError(
                    "recover_interrupted requires a requested tool call"
                )
            if any(
                call.status
                in {ToolStatus.STARTED, ToolStatus.OUTCOME_UNKNOWN}
                for call in turn_calls
            ):
                raise DomainError(
                    "recover_interrupted cannot cover an uncertain tool call"
                )
        elif not any(
            call.status is ToolStatus.OUTCOME_UNKNOWN for call in turn_calls
        ):
            raise DomainError("abandon requires an unknown tool outcome")

        state.pending_recovery_intent = event

    @staticmethod
    def _apply_tool_reconciled(state: SessionState, event: Event) -> None:
        call = SessionReducer._call_for_active_turn(state, event.payload)
        if call.status is not ToolStatus.OUTCOME_UNKNOWN:
            raise DomainError("only an unknown tool outcome can be reconciled")
        raw_outcome = event.payload.get("outcome")
        intent = state.pending_recovery_intent
        if intent is not None and (
            intent.payload["action"] != "abandon"
            or raw_outcome != "abandoned"
        ):
            raise DomainError(
                "tool reconciliation does not match the pending recovery intent"
            )
        statuses = {
            "succeeded": ToolStatus.USER_CONFIRMED_SUCCESS,
            "failed": ToolStatus.USER_CONFIRMED_FAILURE,
            "abandoned": ToolStatus.ABANDONED,
            "user_confirmed_success": ToolStatus.USER_CONFIRMED_SUCCESS,
            "user_confirmed_failure": ToolStatus.USER_CONFIRMED_FAILURE,
        }
        if raw_outcome not in statuses:
            raise DomainError(
                "reconciliation outcome must be succeeded, failed, or abandoned"
            )
        note = event.payload.get("note", "")
        if not isinstance(note, str):
            raise DomainError("reconciliation note must be a string")
        state.tool_calls[call.call_key] = replace(
            call,
            status=statuses[raw_outcome],
            recovery_blocked=False,
            reconciliation_note=note,
            finished_seq=event.seq,
        )
        SessionReducer._refresh_recovery_block(state)

    @staticmethod
    def _apply_turn_finished(state: SessionState, event: Event) -> None:
        turn_id = SessionReducer._require_active_turn(state)
        if state.recovery_blocked:
            raise DomainError(
                "cannot finish a recovery-blocked turn before reconciliation"
            )
        event_turn_id = _payload_uuid(event.payload, "turn_id")
        if event_turn_id != turn_id:
            raise DomainError("turn_finished belongs to another turn")
        status = _turn_status(event.payload.get("status"))
        intent = state.pending_recovery_intent
        if intent is not None:
            expected_status = (
                TurnStatus.INTERRUPTED
                if intent.payload["action"] == "recover_interrupted"
                else TurnStatus.ABANDONED
            )
            if status is not expected_status:
                raise DomainError(
                    "turn status does not match the pending recovery intent"
                )
        unfinished = [
            call.call_key
            for call in state.tool_calls.values()
            if call.turn_id == turn_id and not call.is_terminal
        ]
        if unfinished:
            raise DomainError("cannot finish a turn with unfinished tool calls")
        state.turns[turn_id] = status
        state.active_turn_id = None
        state.recovery_blocked = False
        state.pending_recovery_intent = None

    @staticmethod
    def _apply_compaction_checkpoint(state: SessionState, event: Event) -> None:
        unresolved = any(
            call.status in {
                ToolStatus.REQUESTED,
                ToolStatus.STARTED,
                ToolStatus.OUTCOME_UNKNOWN,
            }
            for call in state.tool_calls.values()
        )
        if unresolved:
            raise DomainError("compaction requires a completed sampling boundary")
        through_seq = event.payload.get("through_seq")
        if type(through_seq) is not int or not 0 <= through_seq < event.seq:
            raise DomainError("through_seq must reference an earlier event")
        for call in state.tool_calls.values():
            if (
                call.requested_seq is not None
                and call.finished_seq is not None
                and call.requested_seq <= through_seq < call.finished_seq
            ):
                raise DomainError(
                    "through_seq must reference a completed sampling boundary"
                )
        summary = event.payload.get("summary")
        if not isinstance(summary, str) or not summary.strip():
            raise DomainError("checkpoint summary must be a non-empty string")
        replacement_conversation = event.payload.get("replacement_conversation")
        if not isinstance(replacement_conversation, (list, tuple)):
            raise DomainError("replacement_conversation must be an array")
        try:
            validate_conversation(replacement_conversation)
        except ConversationError as error:
            raise DomainError(
                f"invalid checkpoint replacement_conversation: {error}"
            ) from error
        if any(
            message.get("role") == "system"
            for message in replacement_conversation
        ):
            raise DomainError(
                "checkpoint replacement_conversation must not contain system messages"
            )
        state.latest_checkpoint = event

    @staticmethod
    def _apply_file_snapshot(state: SessionState, event: Event) -> None:
        SessionReducer._reject_pending_recovery_action(
            state, "record a file snapshot"
        )
        active_turn_id = SessionReducer._require_active_turn(state)
        turn_id = _payload_uuid(event.payload, "turn_id")
        if turn_id != active_turn_id:
            raise DomainError("file snapshot belongs to another turn")
        path = _payload_string(event.payload, "path")
        existed_before = event.payload.get("existed_before")
        if type(existed_before) is not bool:
            raise DomainError("existed_before must be a boolean")
        before_bytes = event.payload.get("before_bytes", "")
        if not isinstance(before_bytes, str):
            raise DomainError("before_bytes must be a string")
        before_encoding = event.payload.get("before_encoding")
        if before_encoding != "base64":
            raise DomainError("before_encoding must be base64")
        before_mode = event.payload.get("before_mode")
        snapshot = FileSnapshot(
            turn_id=turn_id,
            path=path,
            existed_before=existed_before,
            before_bytes=before_bytes,
            before_mode=before_mode,
            source_call_key=event.payload.get("call_key"),
            after_hash=event.payload.get("after_hash"),
            after_mode=event.payload.get("after_mode"),
        )
        call_key = _payload_string(event.payload, "call_key")
        source_call = state.tool_calls.get(call_key)
        if source_call is None:
            raise DomainError(f"unknown file snapshot call_key: {call_key}")
        if source_call.turn_id != active_turn_id:
            raise DomainError("file snapshot call_key is not in the active turn")
        if source_call.status is not ToolStatus.REQUESTED:
            raise DomainError("file snapshot call_key is not an active requested call")
        if source_call.name not in _MANAGED_WRITE_TOOL_NAMES:
            raise DomainError("file snapshot call_key must identify a managed write")
        key = (turn_id, path)
        existing = state.file_snapshots.get(key)
        if existing is None:
            state.file_snapshots[key] = snapshot
        elif existing.after_hash is None and existing.source_call_key is not None:
            source_call = state.tool_calls.get(existing.source_call_key)
            known_no_effect = {
                ToolStatus.FAILED,
                ToolStatus.CONFLICT,
                ToolStatus.INTERRUPTED,
                ToolStatus.CANCELLED,
                ToolStatus.DENIED,
                ToolStatus.INVALID_ARGUMENTS,
                ToolStatus.NOT_EXECUTED,
            }
            if source_call is not None and source_call.status in known_no_effect:
                state.file_snapshots[key] = snapshot

    @staticmethod
    def _apply_sampling_failed(state: SessionState, event: Event) -> None:
        SessionReducer._reject_pending_recovery_action(
            state, "record a sampling failure"
        )
        SessionReducer._require_active_turn(state)

    @staticmethod
    def _apply_undo_finished(state: SessionState, event: Event) -> None:
        turn_id = _payload_uuid(event.payload, "turn_id")
        if turn_id not in state.turns:
            raise DomainError("undo references an unknown turn")
        previous = state.undo_results.get(turn_id)
        if previous is not None and previous.payload.get("status") == "succeeded":
            raise DomainError("turn has already been undone")
        status = event.payload.get("status")
        if status not in {"succeeded", "conflict", "partial"}:
            raise DomainError("undo status must be succeeded, conflict, or partial")
        files = event.payload.get("files")
        if not isinstance(files, (list, tuple)):
            raise DomainError("undo files must be an array")
        allowed_statuses = {
            "restored",
            "deleted",
            "already_restored",
            "already_deleted",
            "conflict",
            "ineligible",
            "not_modified",
            "failed",
        }
        observed_statuses: set[str] = set()
        for item in files:
            if not isinstance(item, Mapping) or set(item) != {"path", "status", "detail"}:
                raise DomainError("each undo file must have exactly path, status, and detail")
            for key in ("path", "status", "detail"):
                if not isinstance(item[key], str):
                    raise DomainError(f"undo file {key} must be a string")
            if item["status"] not in allowed_statuses:
                raise DomainError(f"unknown undo file status: {item['status']!r}")
            observed_statuses.add(item["status"])
        expected_status = reduce_undo_status(observed_statuses)
        if status != expected_status:
            raise DomainError(
                f"undo status {status!r} does not match file results {expected_status!r}"
            )
        state.undo_results[turn_id] = event

    @staticmethod
    def _apply_plan_mode_set(state: SessionState, event: Event) -> None:
        SessionReducer._reject_pending_recovery_action(state, "set plan mode")
        if set(event.payload) != {"active"}:
            raise DomainError("plan_mode_set payload must contain only active")
        active = event.payload.get("active")
        if type(active) is not bool:
            raise DomainError("plan_mode_set active must be a boolean")
        state.plan_mode_active = active

    @staticmethod
    def _require_active_turn(state: SessionState) -> str:
        if state.active_turn_id is None:
            raise DomainError("event requires an active turn")
        return state.active_turn_id

    @staticmethod
    def _reject_pending_recovery_action(
        state: SessionState, action: str
    ) -> None:
        if state.pending_recovery_intent is not None:
            raise DomainError(
                f"cannot {action} while a recovery intent is pending"
            )

    @staticmethod
    def _call_for_active_turn(
        state: SessionState, payload: Mapping[str, Any]
    ) -> ToolCall:
        active_turn_id = SessionReducer._require_active_turn(state)
        call_key = payload.get("call_key")
        if call_key is not None:
            if not isinstance(call_key, str) or not call_key:
                raise DomainError("call_key must be a non-empty string")
            call = state.tool_calls.get(call_key)
            if call is None:
                raise DomainError(f"unknown tool call key: {call_key}")
            provider_call_id = payload.get("call_id")
            if (
                provider_call_id is not None
                and provider_call_id != call.provider_call_id
            ):
                raise DomainError("call_key and call_id identify different calls")
        else:
            provider_call_id = _payload_string(payload, "call_id")
            matches = [
                candidate
                for candidate in state.tool_calls.values()
                if candidate.turn_id == active_turn_id
                and candidate.provider_call_id == provider_call_id
            ]
            if not matches:
                raise DomainError(f"unknown tool call: {provider_call_id}")
            if len(matches) != 1:
                raise DomainError(
                    f"ambiguous tool call id {provider_call_id!r}; call_key is required"
                )
            call = matches[0]
        if call.turn_id != active_turn_id:
            raise DomainError("tool call does not belong to the active turn")
        return call

    @staticmethod
    def _refresh_recovery_block(state: SessionState) -> None:
        unknown = any(
            call.status is ToolStatus.OUTCOME_UNKNOWN
            for call in state.tool_calls.values()
        )
        state.recovery_blocked = unknown
        if state.active_turn_id is not None:
            state.turns[state.active_turn_id] = (
                TurnStatus.RECOVERY_BLOCKED if unknown else TurnStatus.ACTIVE
            )


def reduce_event(state: SessionState, event: Event) -> SessionState:
    """Apply one event to an independent derived copy of ``state``."""

    if not isinstance(state, SessionState) or not isinstance(event, Event):
        raise TypeError("reduce_event requires a SessionState and Event")
    derived = replace(
        state,
        turns=dict(state.turns),
        turn_inputs=dict(state.turn_inputs),
        tool_calls=dict(state.tool_calls),
        file_snapshots=dict(state.file_snapshots),
        undo_results=dict(state.undo_results),
        assistant_events=list(state.assistant_events),
        events=list(state.events),
    )
    return SessionReducer.apply(derived, event)


def plan_recovery_events(state: SessionState) -> list[Event]:
    """Plan explicit terminal facts for calls left open by a crash.

    The function does not mutate state, read the clock, or use randomness.
    Callers append each returned event before applying it to memory.
    """

    if state.session_id is None:
        raise DomainError("cannot plan recovery before session creation")
    if state.last_event_timestamp is None:
        raise DomainError("session state has no replay boundary")

    planned: list[Event] = []
    next_seq = state.last_seq + 1
    namespace = uuid.UUID(state.session_id)
    for call in state.tool_calls.values():
        if call.status is ToolStatus.STARTED:
            status = ToolStatus.OUTCOME_UNKNOWN
            payload: dict[str, Any] = {
                "call_key": call.call_key,
                "call_id": call.provider_call_id,
                "status": status.value,
                "result": "execution began but no terminal result was recorded",
                "recovery_blocked": True,
            }
        elif call.status is ToolStatus.REQUESTED:
            status = ToolStatus.NOT_EXECUTED
            payload = {
                "call_key": call.call_key,
                "call_id": call.provider_call_id,
                "status": status.value,
                "result": "tool call was not started before recovery",
                "recovery_blocked": False,
            }
        else:
            continue
        event_id = str(
            uuid.uuid5(
                namespace,
                f"recovery:tool_finished:{next_seq}:{call.call_key}:{status.value}",
            )
        )
        planned.append(
            Event(
                version=EVENT_VERSION,
                seq=next_seq,
                event_id=event_id,
                timestamp=state.last_event_timestamp,
                session_id=state.session_id,
                type="tool_finished",
                payload=payload,
            )
        )
        next_seq += 1
    return planned
