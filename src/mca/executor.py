"""Durable validation, approval, and execution pipeline for accepted calls."""

from __future__ import annotations

import base64
import hashlib
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from .approval import ApprovalDecision, ApprovalInterrupted, ApprovalRequest, _escape_terminal_text
from .domain import SessionReducer, SessionState, ToolCall, ToolStatus
from .store import RolloutStore
from .tools.filesystem import (
    ExecutedFileChange,
    FileConflictError,
    PathSafetyError,
    PreparedFileChange,
)
from .tools.registry import (
    ExecutionMode,
    SideEffect,
    ToolRegistry,
    ToolResult,
    ToolValidationError,
    UnknownToolError,
)
from .tools.shell import BoundedOutputChannel, PreparedShellCommand


_KNOWN_SECRET_KEYS = (
    "MCA_API_KEY",
    "DEEPSEEK_API_KEY",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
    "AZURE_OPENAI_API_KEY",
    "MISTRAL_API_KEY",
    "GROQ_API_KEY",
)


class Approver(Protocol):
    def decide(self, request: ApprovalRequest) -> ApprovalDecision: ...


class ToolExecutorError(RuntimeError):
    """Raised when durable executor invariants cannot be maintained."""


class PostCommitInterrupted(KeyboardInterrupt):
    """Signal that a file commit succeeded before Ctrl-C was observed."""


@dataclass(frozen=True)
class AcceptedToolCall:
    """The full identity of a call already accepted into SessionState."""

    call_key: str
    provider_call_id: str
    name: str
    raw_arguments: str

    @property
    def call_id(self) -> str:
        return self.provider_call_id

    @property
    def arguments(self) -> str:
        return self.raw_arguments

    @classmethod
    def from_tool_call(cls, call: ToolCall) -> AcceptedToolCall:
        return cls(
            call_key=call.call_key,
            provider_call_id=call.provider_call_id,
            name=call.name,
            raw_arguments=call.arguments,
        )


@dataclass(frozen=True)
class PreparedParallelCall:
    """A started side-effect-free call whose body may run off-thread."""

    call: ToolCall
    name: str
    arguments: dict[str, Any]
    handler: Callable[[dict[str, Any]], object]


class ToolExecutor:
    """Execute each accepted call at most once through durable lifecycle facts."""

    def __init__(
        self,
        registry: ToolRegistry,
        store: RolloutStore,
        state: SessionState,
        approver: Approver,
        workspace: str | os.PathLike[str],
        *,
        output_channel: BoundedOutputChannel | None = None,
    ) -> None:
        workspace_path = Path(workspace).resolve(strict=True)
        if not workspace_path.is_dir():
            raise ValueError("workspace must be a directory")
        if (
            output_channel is not None
            and type(output_channel) is not BoundedOutputChannel
        ):
            raise TypeError(
                "output_channel must be a BoundedOutputChannel or None"
            )
        self.registry = registry
        self.store = store
        self.state = state
        self.approver = approver
        self.workspace = workspace_path
        self.output_channel = output_channel
        self._usable = True
        if registry.workspace is not None and registry.workspace != workspace_path:
            raise ValueError("registry workspace does not match executor workspace")

    def execute(self, accepted: AcceptedToolCall | ToolCall) -> ToolResult:
        if not self._usable:
            raise ToolExecutorError("executor is unusable after reducer divergence")
        call = (
            AcceptedToolCall.from_tool_call(accepted)
            if isinstance(accepted, ToolCall)
            else accepted
        )
        if not isinstance(call, AcceptedToolCall):
            raise TypeError("accepted call must be an AcceptedToolCall or ToolCall")
        state_call = self._require_accepted_identity(call)

        try:
            spec = self.registry.resolve(call.name)
        except UnknownToolError as error:
            return self._finish_requested_error(
                state_call, ToolStatus.UNKNOWN_TOOL, str(error)
            )
        try:
            arguments = self.registry.parse_and_validate(
                call.name, call.raw_arguments
            )
        except ToolValidationError as error:
            return self._finish_requested_error(
                state_call, ToolStatus.INVALID_ARGUMENTS, str(error)
            )

        prepared: object | None = None
        if spec.prepare_handler is not None:
            try:
                prepared = spec.prepare_handler(arguments)
            except Exception as error:
                return self._finish_requested_error(
                    state_call,
                    ToolStatus.INVALID_ARGUMENTS,
                    _safe_error("tool preparation failed", error),
                )

        requires_approval = spec.side_effect not in {SideEffect.NONE, False}
        if spec.side_effect is SideEffect.PLAN_EXIT and not self.state.plan_mode_active:
            # exit_plan_mode stays registered so the tool catalog is stable, but
            # it is only meaningful while plan mode is active.
            return self._finish_requested_error(
                state_call,
                ToolStatus.FAILED,
                "exit_plan_mode is only available while plan mode is active",
            )
        if self.state.plan_mode_active and spec.side_effect in {
            SideEffect.WORKSPACE_WRITE,
            SideEffect.SHELL,
        }:
            # Hard layer: while plan mode is active the runtime refuses every
            # workspace-mutating tool before approval, so the model must research
            # and exit plan mode before it can touch the workspace. exit_plan_mode
            # (PLAN_EXIT) is deliberately exempt so the model can leave plan mode.
            return self._finish_requested_error(
                state_call,
                ToolStatus.DENIED,
                "plan mode is active; call exit_plan_mode to get the plan "
                "approved before running write_file, edit_file, or bash",
            )
        if requires_approval:
            assert prepared is not None
            try:
                request = self._approval_request(spec.side_effect, call.name, prepared)
            except KeyboardInterrupt:
                self._finish_requested_error(
                    state_call,
                    ToolStatus.CANCELLED,
                    "tool approval rendering interrupted",
                )
                raise
            except Exception as error:
                return self._finish_requested_error(
                    state_call,
                    ToolStatus.FAILED,
                    _safe_error("tool approval rendering failed", error),
                )
            if self.state.session_approval_always:
                decision = ApprovalDecision.ALLOW_SESSION
            else:
                try:
                    decision = self._approval_decision(request)
                except ApprovalInterrupted:
                    self._append_and_reduce(
                        "approval_decided",
                        {
                            "call_key": call.call_key,
                            "call_id": call.provider_call_id,
                            "approved": False,
                            "scope": "once",
                        },
                    )
                    self._finish_requested_error(
                        state_call,
                        ToolStatus.CANCELLED,
                        "tool approval interrupted",
                    )
                    raise KeyboardInterrupt from None
            approved = decision in {
                ApprovalDecision.ALLOW_ONCE,
                ApprovalDecision.ALLOW_SESSION,
            }
            scope = (
                "session"
                if decision is ApprovalDecision.ALLOW_SESSION
                else "once"
            )
            self._append_and_reduce(
                "approval_decided",
                {
                    "call_key": call.call_key,
                    "call_id": call.provider_call_id,
                    "approved": approved,
                    "scope": scope,
                },
            )
            if not approved:
                return self._finish_requested_error(
                    state_call,
                    ToolStatus.DENIED,
                    "tool execution denied",
                )

        if isinstance(prepared, PreparedFileChange):
            self._append_mutation_plan(state_call, prepared)
            self._append_first_snapshot(state_call, prepared)

        self._append_and_reduce(
            "tool_started",
            {
                "call_key": call.call_key,
                "call_id": call.provider_call_id,
                "name": call.name,
                "arguments": arguments,
            },
        )
        if spec.side_effect is SideEffect.PLAN_EXIT:
            # Approval of exit_plan_mode is the user's decision to leave plan
            # mode; record it as a durable fact between started and finished so
            # the tool result reflects a completed transition.
            self._append_and_reduce("plan_mode_set", {"active": False})
            result = ToolResult.bounded(
                title="exit_plan_mode",
                output="plan approved; plan mode exited",
            )
            self._finish_started(state_call, result)
            return result
        try:
            if spec.handler is not None:
                raw_result = spec.handler(arguments)
            else:
                assert prepared is not None
                if isinstance(prepared, PreparedShellCommand):
                    raw_result = prepared.execute(output_channel=self.output_channel)
                else:
                    raw_result = prepared.execute()  # type: ignore[attr-defined]
            result = self._normalize_result(call.name, raw_result)
        except (FileConflictError, PathSafetyError) as error:
            # A side-effecting prepared write that fails path/hash revalidation at
            # commit time is a genuine time-of-check/time-of-use conflict. A
            # read-only handler has no prepare step, so the same exception there
            # is just a bad-argument failure and must not borrow conflict's
            # undo/TOCTOU meaning.
            status = (
                ToolStatus.CONFLICT
                if spec.prepare_handler is not None
                else ToolStatus.FAILED
            )
            label = (
                "tool execution conflict"
                if status is ToolStatus.CONFLICT
                else "tool execution failed"
            )
            result = _error_result(call.name, status, _safe_error(label, error))
        except KeyboardInterrupt:
            result = _error_result(
                call.name,
                ToolStatus.INTERRUPTED,
                "tool execution interrupted",
            )
        except Exception as error:
            result = _error_result(
                call.name,
                ToolStatus.FAILED,
                _safe_error("tool execution failed", error),
            )

        try:
            self._finish_started(state_call, result)
        except ToolExecutorError:
            raise
        except Exception as error:
            result = _error_result(
                call.name,
                ToolStatus.FAILED,
                _safe_error("tool result handling failed", error),
            )
            self._finish_started(state_call, result)
        if result.metadata.get("interruption_warning") is True:
            raise PostCommitInterrupted from None
        return result

    def execute_call(self, accepted: AcceptedToolCall | ToolCall) -> ToolResult:
        """Compatibility spelling for callers that name the unit explicitly."""

        return self.execute(accepted)

    def prepare_parallel(
        self, accepted: AcceptedToolCall | ToolCall
    ) -> PreparedParallelCall:
        """Validate and durably start one explicitly safe handler call."""

        if not self._usable:
            raise ToolExecutorError("executor is unusable after reducer divergence")
        call = (
            AcceptedToolCall.from_tool_call(accepted)
            if isinstance(accepted, ToolCall)
            else accepted
        )
        if not isinstance(call, AcceptedToolCall):
            raise TypeError("accepted call must be an AcceptedToolCall or ToolCall")
        state_call = self._require_accepted_identity(call)
        if (
            self.registry.execution_mode(call.name, call.raw_arguments)
            is not ExecutionMode.PARALLEL
        ):
            raise ToolExecutorError(
                f"accepted call is not concurrency-safe: {call.call_key}"
            )
        spec = self.registry.resolve(call.name)
        arguments = self.registry.parse_and_validate(
            call.name, call.raw_arguments
        )
        if spec.handler is None:
            raise ToolExecutorError(
                f"parallel call has no direct handler: {call.call_key}"
            )
        self._append_and_reduce(
            "tool_started",
            {
                "call_key": call.call_key,
                "call_id": call.provider_call_id,
                "name": call.name,
                "arguments": arguments,
            },
        )
        return PreparedParallelCall(
            call=state_call,
            name=call.name,
            arguments=arguments,
            handler=spec.handler,
        )

    def dispatch_parallel(self, prepared: PreparedParallelCall) -> ToolResult:
        """Run only a prepared safe handler body without touching session state."""

        if not isinstance(prepared, PreparedParallelCall):
            raise TypeError("prepared must be a PreparedParallelCall")
        try:
            raw_result = prepared.handler(prepared.arguments)
            return self._normalize_result(prepared.name, raw_result)
        except (FileConflictError, PathSafetyError) as error:
            return _error_result(
                prepared.name,
                ToolStatus.FAILED,
                _safe_error("tool execution failed", error),
            )
        except KeyboardInterrupt:
            return _error_result(
                prepared.name,
                ToolStatus.INTERRUPTED,
                "tool execution interrupted",
            )
        except Exception as error:
            return _error_result(
                prepared.name,
                ToolStatus.FAILED,
                _safe_error("tool execution failed", error),
            )

    def commit_parallel(
        self, prepared: PreparedParallelCall, result: ToolResult
    ) -> None:
        """Persist one settled safe result on the caller's serial path."""

        if not isinstance(prepared, PreparedParallelCall):
            raise TypeError("prepared must be a PreparedParallelCall")
        state_call = self.state.tool_calls.get(prepared.call.call_key)
        if state_call is None or state_call.status is not ToolStatus.STARTED:
            raise ToolExecutorError(
                f"parallel call is not started: {prepared.call.call_key}"
            )
        self._finish_started(state_call, result)

    def _require_accepted_identity(self, accepted: AcceptedToolCall) -> ToolCall:
        state_call = self.state.tool_calls.get(accepted.call_key)
        if state_call is None:
            raise ToolExecutorError(f"unknown accepted call: {accepted.call_key}")
        identity = (
            state_call.provider_call_id,
            state_call.name,
            state_call.arguments,
        )
        supplied = (
            accepted.provider_call_id,
            accepted.name,
            accepted.raw_arguments,
        )
        if identity != supplied:
            raise ToolExecutorError(
                f"accepted call identity mismatch: {accepted.call_key}"
            )
        if state_call.status is not ToolStatus.REQUESTED:
            raise ToolExecutorError(
                f"accepted call is not pending: {accepted.call_key}"
            )
        if self.state.active_turn_id != state_call.turn_id:
            raise ToolExecutorError("accepted call is not in the active turn")
        return state_call

    def _approval_decision(self, request: ApprovalRequest) -> ApprovalDecision:
        try:
            decision = self.approver.decide(request)
        except ApprovalInterrupted:
            raise
        except KeyboardInterrupt:
            raise ApprovalInterrupted from None
        except Exception:
            return ApprovalDecision.DENY
        if decision in {
            ApprovalDecision.ALLOW_ONCE,
            ApprovalDecision.ALLOW_SESSION,
        }:
            return decision
        return ApprovalDecision.DENY

    def _approval_request(
        self, side_effect: SideEffect | bool, name: str, prepared: object
    ) -> ApprovalRequest:
        if side_effect is SideEffect.WORKSPACE_WRITE and isinstance(
            prepared, PreparedFileChange
        ):
            return ApprovalRequest.for_file(name, prepared)
        if side_effect is SideEffect.SHELL and isinstance(
            prepared, PreparedShellCommand
        ):
            return ApprovalRequest.for_shell(
                command=prepared.command, cwd=prepared.cwd
            )
        spec = self.registry.resolve(name)
        if spec.approval_renderer is not None:
            rendered = spec.approval_renderer(prepared)
            if not isinstance(rendered, str):
                raise ToolExecutorError("approval renderer must return a string")
            return ApprovalRequest(
                tool_name=name,
                target=_escape_terminal_text(rendered),
                kind="rendered",
            )
        raise ToolExecutorError(f"tool has no supported approval target: {name}")

    def _append_first_snapshot(
        self, call: ToolCall, prepared: PreparedFileChange
    ) -> None:
        path = str(prepared.canonical_path)
        existing = self.state.file_snapshots.get((call.turn_id, path))
        if existing is not None and existing.after_hash is not None:
            return
        self._append_and_reduce(
            "file_snapshot",
            {
                "turn_id": call.turn_id,
                "path": path,
                "existed_before": prepared.existed_before,
                "before_bytes": base64.b64encode(prepared.before_bytes).decode("ascii"),
                "before_encoding": "base64",
                "before_mode": prepared.before_mode,
                "before_hash": prepared.before_hash,
                "call_key": call.call_key,
            },
        )

    def _append_mutation_plan(
        self, call: ToolCall, prepared: PreparedFileChange
    ) -> None:
        self._append_and_reduce(
            "file_mutation_planned",
            {
                "turn_id": call.turn_id,
                "call_key": call.call_key,
                "path": str(prepared.canonical_path),
                "expected_version": prepared.expected_version.to_dict(),
                "proposed_hash": hashlib.sha256(
                    prepared.proposed_bytes
                ).hexdigest(),
                "diff": prepared.diff,
            },
        )

    def _finish_requested_error(
        self, call: ToolCall, status: ToolStatus, message: str
    ) -> ToolResult:
        result = _error_result(call.name, status, message)
        self._append_and_reduce(
            "tool_finished",
            {
                "call_key": call.call_key,
                "call_id": call.provider_call_id,
                "status": status.value,
                "result": result.output,
                "truncated": bool(result.metadata["truncated"]),
            },
        )
        return result

    def _finish_started(self, call: ToolCall, result: ToolResult) -> None:
        try:
            status = ToolStatus(result.status)
        except ValueError:
            status = ToolStatus.FAILED
            result = _error_result(
                call.name, status, f"tool returned invalid status: {result.status}"
            )
        if status not in {
            ToolStatus.SUCCEEDED,
            ToolStatus.FAILED,
            ToolStatus.TIMED_OUT,
            ToolStatus.INTERRUPTED,
            ToolStatus.CONFLICT,
            ToolStatus.CANCELLED,
        }:
            status = ToolStatus.FAILED
            result = _error_result(
                call.name, status, f"tool returned invalid status: {result.status}"
            )
        payload: dict[str, Any] = {
            "call_key": call.call_key,
            "call_id": call.provider_call_id,
            "status": status.value,
            "result": result.output,
            "truncated": bool(result.metadata.get("truncated", False)),
        }
        exit_code = result.metadata.get("exit_code")
        if type(exit_code) is int:
            payload["exit_code"] = exit_code
        path = result.metadata.get("path")
        after_hash = result.metadata.get("after_hash")
        after_mode = result.metadata.get("after_mode")
        after_version = result.metadata.get("after_version")
        if status is ToolStatus.SUCCEEDED and isinstance(path, str) and isinstance(
            after_hash, str
        ) and type(after_mode) is int:
            payload["path"] = path
            payload["after_hash"] = after_hash
            payload["after_mode"] = after_mode
            if isinstance(after_version, dict):
                payload["after_version"] = after_version
            created = result.metadata.get("created_directories")
            if isinstance(created, tuple) and created:
                payload["created_directories"] = list(created)
        self._append_and_reduce("tool_finished", payload)

    def _normalize_result(self, name: str, raw_result: object) -> ToolResult:
        if isinstance(raw_result, ToolResult):
            if (
                not isinstance(raw_result.title, str)
                or not raw_result.title
                or not isinstance(raw_result.output, str)
                or not isinstance(raw_result.status, str)
                or not hasattr(raw_result.metadata, "get")
            ):
                raise TypeError(f"{name} returned an invalid ToolResult")
            return raw_result
        if isinstance(raw_result, ExecutedFileChange):
            path = str(raw_result.canonical_path)
            return ToolResult.bounded(
                title=f"Write {path}",
                output=f"wrote {path}",
                metadata={
                    "path": path,
                    "before_hash": raw_result.before_hash,
                    "after_hash": raw_result.after_hash,
                    "after_mode": raw_result.after_mode,
                    "after_version": raw_result.after_version.to_dict(),
                    "created_directories": raw_result.created_directories,
                    "durability_warning": raw_result.durability_warning,
                    "interruption_warning": raw_result.interruption_warning,
                },
            )
        raise TypeError(f"{name} returned an unsupported result")

    def _append_and_reduce(
        self, event_type: str, payload: dict[str, Any]
    ) -> None:
        event = self.store.append(event_type, payload)
        try:
            SessionReducer.apply(self.state, event)
        except Exception as error:
            self._usable = False
            raise ToolExecutorError(
                f"durable event {event.seq} could not be applied to state"
            ) from error


def _error_result(name: str, status: ToolStatus, message: str) -> ToolResult:
    return ToolResult.bounded(
        title=f"{name} error",
        output=message,
        status=status.value,
    )


def _safe_error(prefix: str, error: BaseException) -> str:
    detail = str(error)
    for key in _KNOWN_SECRET_KEYS:
        secret = os.environ.get(key)
        if secret:
            detail = detail.replace(secret, "<redacted>")
    if not detail:
        detail = type(error).__name__
    return f"{prefix}: {type(error).__name__}: {detail}"


__all__ = [
    "AcceptedToolCall",
    "PreparedParallelCall",
    "PostCommitInterrupted",
    "ToolExecutor",
    "ToolExecutorError",
]
