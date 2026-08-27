"""Command-line interface and REPL that assemble the mca runtime.

The CLI is only glue: it reads configuration, opens or resumes a rollout,
builds the tool registry, approver, model client, and AgentLoop, then drives
one turn (one-shot) or many turns (REPL). All durable facts and recovery
semantics live in the runtime modules; this file never invents state.
"""

from __future__ import annotations

import argparse
import dataclasses
import platform as platform_module
import sys
import uuid
from collections.abc import Sequence
from datetime import date as _date
from pathlib import Path

from .agent import AgentLoop, RecoveryBlockedError
from .approval import InteractiveApprover
from .compact import CompactionError, SessionCompactor
from .config import Config
from .domain import SessionReducer, SessionState, ToolStatus, TurnStatus
from .executor import ToolExecutor
from .inspect import list_session_ids, render_transcript, summarize
from .model import ModelClient
from .projection import ProjectionEnvironment, PromptProjector, estimate_request_tokens
from .session import (
    ReconciliationError,
    ResumeError,
    ResumedSession,
    continuable_turn_id,
    reconcile_tool,
    resume_session,
)
from .store import RolloutCorruptionError, RolloutStore, SessionLockedError
from .tools import create_tool_registry
from .undo import ManagedUndo, UndoError


SESSIONS_ROOT = Path(".mca") / "sessions"
_SUCCESS_STATUSES = {TurnStatus.COMPLETED, TurnStatus.MAX_STEPS_REACHED}
_REPL_HELP = (
    "Commands:\n"
    "  /help     show this help\n"
    "  /status   show session summary and context budget\n"
    "  /compact  compact the conversation into a checkpoint\n"
    "  /undo     undo the managed file writes of the last finished turn\n"
    "  /exit     leave the REPL\n"
    "Any other line is sent to the agent as a new task."
)


def _sessions_root(workspace: Path) -> Path:
    return workspace / SESSIONS_ROOT


def _live_environment(workspace: Path) -> ProjectionEnvironment:
    is_git = any(
        (parent / ".git").exists() for parent in (workspace, *workspace.parents)
    )
    return ProjectionEnvironment(
        cwd=str(workspace),
        platform=platform_module.system() or "unknown",
        date=_date.today().isoformat(),
        is_git=is_git,
    )


class _Console:
    """Stdout writer that also renders streamed assistant content."""

    def __init__(self, *, verbose: bool) -> None:
        self.verbose = verbose
        self._streaming = False

    def line(self, text: str = "") -> None:
        if self._streaming:
            sys.stdout.write("\n")
            self._streaming = False
        print(text)

    def stream(self, delta: str) -> None:
        if not delta:
            return
        sys.stdout.write(delta)
        sys.stdout.flush()
        self._streaming = True

    def invalidate(self) -> None:
        if self._streaming:
            sys.stdout.write("\n[output discarded]\n")
            sys.stdout.flush()
            self._streaming = False


class _Runtime:
    """A locked store, its derived state, and a ready AgentLoop."""

    def __init__(
        self,
        *,
        store: RolloutStore,
        state: SessionState,
        workspace: Path,
        config: Config,
        console: _Console,
    ) -> None:
        self.store = store
        self.state = state
        self.workspace = workspace
        self.config = config
        self.console = console
        self.model = ModelClient(config)
        registry = create_tool_registry(workspace)
        approver = InteractiveApprover(yolo=config.yolo)
        self.executor = ToolExecutor(
            registry=registry,
            store=store,
            state=state,
            approver=approver,
            workspace=workspace,
        )
        self.compactor = SessionCompactor(
            store=store,
            state=state,
            model=self.model,
            environment=lambda: _live_environment(workspace),
        )
        self.loop = AgentLoop(
            config=config,
            store=store,
            state=state,
            model=self.model,
            executor=self.executor,
            environment=lambda: _live_environment(workspace),
            compactor=self.compactor,
            on_content=console.stream,
            on_invalidate=console.invalidate,
        )

    def close(self) -> None:
        self.model.close()
        self.store.close()

    def report(self, result: object) -> None:
        status = getattr(result, "status", None)
        final_text = getattr(result, "final_text", "")
        error = getattr(result, "error", None)
        if final_text:
            self.console.line(final_text)
        label = status.value if isinstance(status, TurnStatus) else str(status)
        if status in _SUCCESS_STATUSES:
            if self.console.verbose:
                self.console.line(f"[turn {label}]")
        else:
            self.console.line(f"[turn {label}: {error or 'no result'}]")

    def undo_last_turn(self) -> None:
        turn_id = _last_finished_turn(self.state)
        if turn_id is None:
            self.console.line("[undo: no finished turn to undo]")
            return
        undo = ManagedUndo(self.store, self.state, self.workspace)
        try:
            result = undo.undo_turn(turn_id)
        except UndoError as error:
            self.console.line(f"[undo failed: {error}]")
            return
        self.console.line(f"[undo {result.status}]")
        for item in result.files:
            self.console.line(f"  {item.status}: {item.path}")

    def compact_now(self) -> None:
        try:
            self.compactor.compact()
        except CompactionError as error:
            self.console.line(f"[compact skipped: {error}]")
            return
        self.console.line("[compacted conversation into a checkpoint]")

    def status(self) -> None:
        summary = summarize(self.state)
        self.console.line(summary.render_line())
        try:
            messages = PromptProjector.project(
                self.store.load(), self.state, _live_environment(self.workspace)
            )
            schemas = self.executor.registry.provider_schemas()
            tokens = estimate_request_tokens(messages, schemas)
            self.console.line(
                f"[context ~{tokens} tokens / {self.config.context_window} window]"
            )
        except Exception:
            self.console.line("[context estimate unavailable at this boundary]")


def _last_finished_turn(state: SessionState) -> str | None:
    for event in reversed(state.events):
        if event.type == "turn_finished":
            turn_id = event.payload.get("turn_id")
            if isinstance(turn_id, str):
                return turn_id
    return None


def _create_session(
    workspace: Path, config: Config, console: _Console
) -> _Runtime:
    session_id = str(uuid.uuid4())
    store = RolloutStore.create(_sessions_root(workspace), session_id)
    state = SessionState()
    event = store.append(
        "session_created",
        {
            "cwd": str(workspace),
            "model": config.model,
            "context_window": config.context_window,
        },
    )
    SessionReducer.apply(state, event)
    console.line(f"[session {session_id}]")
    return _Runtime(
        store=store,
        state=state,
        workspace=workspace,
        config=config,
        console=console,
    )


def _resume_session(
    workspace: Path, session_id: str, config: Config, console: _Console
) -> _Runtime:
    resumed: ResumedSession = resume_session(
        _sessions_root(workspace), session_id, workspace
    )
    runtime = _Runtime(
        store=resumed.store,
        state=resumed.state,
        workspace=workspace,
        config=config,
        console=console,
    )
    console.line(f"[resumed session {session_id}]")
    _reconcile_if_blocked(runtime)
    return runtime


def _reconcile_if_blocked(runtime: _Runtime) -> None:
    """Walk the operator through unknown tool outcomes left by a crash."""

    state = runtime.state
    console = runtime.console
    while state.recovery_blocked:
        call = next(
            (
                tool_call
                for tool_call in state.tool_calls.values()
                if tool_call.status is ToolStatus.OUTCOME_UNKNOWN
            ),
            None,
        )
        if call is None:
            break
        console.line(
            "[recovery] a tool call's outcome is unknown after an earlier crash:"
        )
        console.line(f"  tool: {call.name}  args: {call.arguments}")
        console.line("  inspect the real workspace, then choose an outcome.")
        try:
            answer = input(
                "  [s]ucceeded / [f]ailed / [a]bandon turn: "
            ).strip().lower()
        except (EOFError, KeyboardInterrupt):
            console.line("\n[recovery deferred; session remains blocked]")
            return
        outcome = {"s": "succeeded", "f": "failed", "a": "abandoned"}.get(answer)
        if outcome is None:
            console.line("  please answer s, f, or a.")
            continue
        try:
            reconcile_tool(runtime.store, state, call.call_key, outcome)
        except ReconciliationError as error:
            console.line(f"[recovery error: {error}]")
            return


def _run_once(runtime: _Runtime, prompt: str) -> int:
    result = runtime.loop.run_turn(prompt)
    runtime.report(result)
    return 0 if result.status in _SUCCESS_STATUSES else 1


def _repl(runtime: _Runtime) -> int:
    console = runtime.console
    console.line("mca REPL. Type /help for commands, /exit to quit.")
    try:
        pending_turn = continuable_turn_id(runtime.state)
    except ReconciliationError:
        pending_turn = None
    if pending_turn is not None:
        console.line("[continuing the recovered turn]")
        runtime.report(runtime.loop.resume_active_turn())
    while True:
        try:
            line = input("mca> ")
        except EOFError:
            console.line()
            return 0
        except KeyboardInterrupt:
            console.line("\n[interrupted; type /exit to quit]")
            continue
        command = line.strip()
        if not command:
            continue
        if command in {"/exit", "/quit"}:
            return 0
        if command == "/help":
            console.line(_REPL_HELP)
            continue
        if command == "/status":
            runtime.status()
            continue
        if command == "/compact":
            runtime.compact_now()
            continue
        if command == "/undo":
            runtime.undo_last_turn()
            continue
        if command.startswith("/"):
            console.line(f"[unknown command: {command}]")
            continue
        try:
            runtime.report(runtime.loop.run_turn(command))
        except RecoveryBlockedError:
            console.line("[session is blocked on an unknown tool outcome]")
            _reconcile_if_blocked(runtime)


def _list_sessions(workspace: Path, console: _Console) -> int:
    """Print a one-line digest per session without taking any lock."""

    sessions_root = _sessions_root(workspace)
    session_ids = list_session_ids(sessions_root)
    if not session_ids:
        console.line("[no sessions in this workspace]")
        return 0
    for session_id in session_ids:
        try:
            events = RolloutStore.read_session_snapshot(sessions_root, session_id)
            state = SessionReducer.replay(events)
            console.line(summarize(state).render_line())
        except (RolloutCorruptionError, ValueError) as error:
            console.line(f"{session_id}  [unreadable: {error}]")
    return 0


def _show_session(workspace: Path, session_id: str, console: _Console) -> int:
    """Print a session transcript without taking any lock."""

    sessions_root = _sessions_root(workspace)
    try:
        events = RolloutStore.read_session_snapshot(sessions_root, session_id)
    except FileNotFoundError:
        console.line(f"[error: session {session_id!r} does not exist]")
        return 1
    except ValueError as error:
        console.line(f"[error: {error}]")
        return 1
    try:
        state = SessionReducer.replay(events)
    except Exception as error:
        console.line(f"[error: session is corrupt: {error}]")
        return 1
    console.line(render_transcript(state))
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mca",
        description="Run the mca coding agent over the current workspace.",
    )
    parser.add_argument(
        "prompt",
        nargs="?",
        help="the task to run once; omit to enter the REPL",
    )
    parser.add_argument(
        "--resume",
        metavar="SESSION_ID",
        help="resume a previous session by its UUID",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="list sessions in the workspace and exit (read-only)",
    )
    parser.add_argument(
        "--show",
        metavar="SESSION_ID",
        help="print a session transcript and exit (read-only)",
    )
    parser.add_argument(
        "--workspace",
        default=None,
        help="workspace directory (default: current directory)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="stream assistant output and show turn status",
    )
    parser.add_argument(
        "--yolo",
        action="store_true",
        help="skip interactive approval (path and command checks stay on)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Parse arguments, assemble the runtime, and drive one session."""

    args = _build_parser().parse_args(argv)
    console = _Console(verbose=args.verbose)

    try:
        workspace = Path(args.workspace or Path.cwd()).resolve(strict=True)
    except (FileNotFoundError, OSError):
        console.line("[error: workspace does not exist]")
        return 1
    if not workspace.is_dir():
        console.line("[error: workspace must be a directory]")
        return 1

    if args.list:
        return _list_sessions(workspace, console)
    if args.show is not None:
        return _show_session(workspace, args.show, console)

    try:
        base_config = Config.from_env()
    except ValueError as error:
        console.line(f"[error: {error}]")
        return 1
    config = dataclasses.replace(
        base_config, verbose=args.verbose, yolo=args.yolo
    )

    if args.yolo:
        console.line(
            "[yolo: interactive approval is OFF; tools run without confirmation]"
        )

    try:
        if args.resume is not None:
            runtime = _resume_session(workspace, args.resume, config, console)
        else:
            runtime = _create_session(workspace, config, console)
    except ValueError as error:
        console.line(f"[error: {error}]")
        return 1
    except SessionLockedError:
        console.line("[error: session is already open in another process]")
        return 1
    except ResumeError as error:
        console.line(f"[resume failed: {error}]")
        return 1

    try:
        if runtime.state.recovery_blocked:
            console.line("[session is blocked until recovery is reconciled]")
            return 1
        if args.prompt is not None:
            return _run_once(runtime, args.prompt)
        return _repl(runtime)
    except KeyboardInterrupt:
        console.line("\n[interrupted]")
        return 130
    finally:
        runtime.close()


__all__ = ["main"]
