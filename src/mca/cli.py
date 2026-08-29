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
from .approval import InteractiveApprover, _escape_terminal_text
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
from .terminal import TerminalInputError, TerminalTheme, read_multiline_prompt
from .tools import create_tool_registry
from .undo import ManagedUndo, UndoError


SESSIONS_ROOT = Path(".mca") / "sessions"
_SUCCESS_STATUSES = {TurnStatus.COMPLETED, TurnStatus.MAX_STEPS_REACHED}
_REPL_HELP = (
    "Commands:\n"
    "  /help     show this help\n"
    "  /status   show session summary and context budget\n"
    "  /plan     enter plan mode (research first; writes are blocked)\n"
    "  /plan off leave plan mode\n"
    "  /compact  compact the conversation into a checkpoint\n"
    "  /undo     undo the managed file writes of the last finished turn\n"
    "  /approval reset  return from session always approval to prompts\n"
    "  /exit     leave the REPL\n"
    "Enter inserts a new line. Ctrl+Enter submits; Ctrl+S is the fallback."
)
_UNBOUND_REPL_HELP = (
    "Before the first task, bind mca to a project with:\n"
    "  workspace: /absolute/path/to/project | your task\n"
    "Commands before binding: /help, /plan, /plan off, /exit.\n"
    "Enter inserts a new line. Ctrl+Enter submits; Ctrl+S is the fallback.\n"
    "The selected directory becomes the only workspace for this session."
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

    def __init__(self, *, verbose: bool, color: bool | None = None) -> None:
        self.verbose = verbose
        self._streaming = False
        self._reasoning_streaming = False
        self._streamed_assistant_text = ""
        self._separate_next_live_block = False
        enabled = sys.stdout.isatty() if color is None else color
        self.theme = TerminalTheme.auto(isatty=enabled)

    def line(self, text: str = "", *, role: str | None = None) -> None:
        if self._streaming:
            sys.stdout.write("\n")
            self._streaming = False
        print(self.theme.style(text, role) if role is not None else text)

    def badge(self, label: str, text: str, *, role: str) -> None:
        self.line(
            f"{self.theme.style('[' + label + ']', role)} {text}"
        )

    def approval(self, text: str) -> None:
        self._start_live_block()
        self.badge("approval", text, role="approval")
        self._separate_next_live_block = True

    def _start_live_block(self) -> None:
        if self._reasoning_streaming or self._streaming:
            sys.stdout.write("\n\n")
            self._reasoning_streaming = False
            self._streaming = False
        elif self._separate_next_live_block:
            sys.stdout.write("\n")
        self._separate_next_live_block = False

    def stream(self, delta: str) -> None:
        if not delta:
            return
        if not self._streaming:
            self._start_live_block()
        sys.stdout.write(self.theme.style(delta, "model"))
        sys.stdout.flush()
        self._streaming = True
        self._streamed_assistant_text += delta

    def invalidate(self) -> None:
        if self._streaming:
            sys.stdout.write("\n" + self.theme.style("[output discarded]\n", "failure"))
            sys.stdout.flush()
            self._streaming = False
        self._streamed_assistant_text = ""

    def reasoning(self, delta: str) -> None:
        """Show only provider-supplied reasoning deltas in a muted stream."""

        if not self.verbose or not delta:
            return
        if not self._reasoning_streaming:
            self._start_live_block()
            sys.stdout.write(self.theme.style("[thinking] ", "muted"))
            self._reasoning_streaming = True
        rendered = _escape_terminal_text(delta, preserve_newlines=True)
        sys.stdout.write(self.theme.style(rendered, "muted"))
        sys.stdout.flush()

    def tool_calls(self, calls: Sequence[object]) -> None:
        self._start_live_block()
        self._streamed_assistant_text = ""
        for call in calls:
            name = _escape_terminal_text(str(getattr(call, "name", "tool")))
            arguments = _escape_terminal_text(str(getattr(call, "arguments", "")))
            if len(arguments) > 220:
                arguments = arguments[:200] + " ... [truncated]"
            self.badge("tool call", f"{name} {arguments}", role="tool")
        self._separate_next_live_block = bool(calls)

    def final_text_was_streamed(self, final_text: str) -> bool:
        streamed = bool(final_text) and self._streamed_assistant_text == final_text
        self._streamed_assistant_text = ""
        if streamed and self._streaming:
            sys.stdout.write("\n")
            sys.stdout.flush()
            self._streaming = False
        return streamed


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
        approver = InteractiveApprover(
            yolo=config.yolo,
            output_fn=console.approval,
            input_fn=lambda prompt: input(console.theme.style(prompt, "approval")),
        )
        self.approver = approver
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
            on_reasoning=console.reasoning,
            on_tool_calls=console.tool_calls,
        )

    def close(self) -> None:
        self.model.close()
        self.store.close()

    def report(self, result: object) -> None:
        status = getattr(result, "status", None)
        final_text = getattr(result, "final_text", "")
        error = getattr(result, "error", None)
        if final_text and not self.console.final_text_was_streamed(final_text):
            self.console.line(final_text)
        label = status.value if isinstance(status, TurnStatus) else str(status)
        if status in _SUCCESS_STATUSES:
            if self.console.verbose:
                self.console.badge("turn", label, role="success")
        else:
            self.console.badge(
                "turn", f"{label}: {error or 'no result'}", role="failure"
            )

    def undo_last_turn(self) -> None:
        turn_id = _last_finished_turn(self.state)
        if turn_id is None:
            self.console.badge("undo", "no finished turn to undo", role="muted")
            return
        undo = ManagedUndo(self.store, self.state, self.workspace)
        try:
            result = undo.undo_turn(turn_id)
        except UndoError as error:
            self.console.badge("undo", f"failed: {error}", role="failure")
            return
        self.console.badge("undo", result.status, role="success")
        for item in result.files:
            self.console.line(f"  {item.status}: {item.path}")

    def compact_now(self) -> None:
        try:
            self.compactor.compact()
        except CompactionError as error:
            self.console.badge("compact", f"skipped: {error}", role="approval")
            return
        self.console.badge("compact", "conversation checkpoint created", role="info")

    def status(self) -> None:
        summary = summarize(self.state)
        self.console.line(summary.render_line())
        if self.state.plan_mode_active:
            self.console.badge("plan", "plan mode: ON — writes are blocked", role="approval")
        try:
            messages = PromptProjector.project(
                self.store.load(), self.state, _live_environment(self.workspace)
            )
            schemas = self.executor.registry.provider_schemas()
            tokens = estimate_request_tokens(messages, schemas)
            self.console.badge(
                "context",
                f"~{tokens} tokens / {self.config.context_window} window",
                role="muted",
            )
        except Exception:
            self.console.badge("context", "estimate unavailable at this boundary", role="muted")

    def set_plan_mode(self, active: bool) -> None:
        if self.state.plan_mode_active == active:
            self.console.badge("plan", f"already {'on' if active else 'off'}", role="muted")
            return
        event = self.store.append("plan_mode_set", {"active": active})
        SessionReducer.apply(self.state, event)
        self.console.badge("plan", f"plan mode {'on' if active else 'off'}", role="approval" if active else "info")

    def reset_session_approval(self) -> None:
        if not self.state.session_approval_always:
            self.console.badge("approval", "already prompting for every side effect", role="muted")
            return
        event = self.store.append("session_approval_reset", {})
        SessionReducer.apply(self.state, event)
        self.approver.reset_session_approval()
        self.console.badge("approval", "session always approval reset", role="approval")


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
    console.badge("session", session_id, role="info")
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
    console.badge("session", f"resumed session {session_id}", role="info")
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


def _read_repl_prompt(console: _Console) -> str:
    """Use the raw editor on a real TTY, retain line input for automation."""

    try:
        return read_multiline_prompt(
            prompt=console.theme.style("mca> ", "prompt"),
            continuation=console.theme.style("...  ", "muted"),
        )
    except TerminalInputError:
        # Unit tests, piped CLI use, and dumb terminals retain the original
        # one-line input contract. The main interactive experience is raw mode.
        return input("mca> ")


def _parse_workspace_prompt(line: str) -> tuple[Path, str]:
    """Parse the one allowed unbound-REPL task header.

    Binding is explicit rather than inferred from prose so the task's durable
    cwd, file-tool boundary, shell cwd, and undo boundary all agree before the
    first event is appended.
    """

    prefix = "workspace:"
    if not line.startswith(prefix):
        raise ValueError(
            "workspace required: start with workspace: /absolute/path | task"
        )
    raw = line[len(prefix) :].strip()
    path_text, separator, task = raw.partition("|")
    if not separator or not path_text.strip() or not task.strip():
        raise ValueError(
            "workspace prompt must be workspace: /absolute/path | task"
        )
    raw_path = Path(path_text.strip())
    if not raw_path.is_absolute():
        raise ValueError("workspace must be an absolute path")
    try:
        workspace = raw_path.resolve(strict=True)
    except (FileNotFoundError, OSError):
        raise ValueError("workspace does not exist") from None
    if not workspace.is_dir():
        raise ValueError("workspace must be a directory")
    return workspace, task.strip()


def _unbound_repl(
    config: Config, console: _Console, *, plan_requested: bool
) -> int:
    """Wait for an explicit workspace before creating any session or tools."""

    requested_plan = plan_requested
    console.badge("mca", "bind a project before the first task; type /help", role="info")
    while True:
        try:
            line = _read_repl_prompt(console)
        except EOFError:
            console.line()
            return 0
        except KeyboardInterrupt:
            console.badge("input", "interrupted; type /exit to quit", role="muted")
            continue
        command = line.strip()
        if not command:
            continue
        if command in {"/exit", "/quit"}:
            return 0
        if command == "/help":
            console.line(_UNBOUND_REPL_HELP)
            continue
        if command == "/plan" or command == "/plan on":
            requested_plan = True
            console.badge("plan", "will turn on when a workspace is bound", role="approval")
            continue
        if command == "/plan off":
            requested_plan = False
            console.badge("plan", "will remain off when a workspace is bound", role="muted")
            continue
        if command.startswith("/"):
            console.badge("command", f"unknown before workspace binding: {command}", role="failure")
            continue
        try:
            workspace, task = _parse_workspace_prompt(command)
        except ValueError as error:
            console.badge("error", str(error), role="failure")
            continue
        runtime = _create_session(workspace, config, console)
        try:
            if requested_plan:
                runtime.set_plan_mode(True)
            console.badge("workspace", f"bound to {workspace}", role="workspace")
            runtime.report(runtime.loop.run_turn(task))
            return _repl(runtime)
        finally:
            runtime.close()


def _repl(runtime: _Runtime) -> int:
    console = runtime.console
    console.badge("mca", "ready — /help lists commands", role="info")
    try:
        pending_turn = continuable_turn_id(runtime.state)
    except ReconciliationError:
        pending_turn = None
    if pending_turn is not None:
        console.badge("recovery", "continuing recovered turn", role="approval")
        runtime.report(runtime.loop.resume_active_turn())
    while True:
        try:
            line = _read_repl_prompt(console)
        except EOFError:
            console.line()
            return 0
        except KeyboardInterrupt:
            console.badge("input", "interrupted; type /exit to quit", role="muted")
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
        if command == "/plan" or command == "/plan on":
            runtime.set_plan_mode(True)
            continue
        if command == "/plan off":
            runtime.set_plan_mode(False)
            continue
        if command == "/compact":
            runtime.compact_now()
            continue
        if command == "/undo":
            runtime.undo_last_turn()
            continue
        if command == "/approval reset":
            runtime.reset_session_approval()
            continue
        if command.startswith("/"):
            console.badge("command", f"unknown: {command}", role="failure")
            continue
        if command.startswith("workspace:"):
            console.badge(
                "workspace",
                "binding is only valid before the first turn; this session is already bound",
                role="approval",
            )
            continue
        try:
            runtime.report(runtime.loop.run_turn(command))
        except RecoveryBlockedError:
            console.badge("recovery", "session is blocked on an unknown tool outcome", role="failure")
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
    parser.add_argument(
        "--plan",
        action="store_true",
        help="start in plan mode: research first; writes are blocked until approved",
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

    if args.prompt is None and args.resume is None and args.workspace is None:
        return _unbound_repl(config, console, plan_requested=args.plan)

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
        if args.plan and not runtime.state.plan_mode_active:
            runtime.set_plan_mode(True)
        if args.prompt is not None:
            return _run_once(runtime, args.prompt)
        return _repl(runtime)
    except KeyboardInterrupt:
        console.line("\n[interrupted]")
        return 130
    finally:
        runtime.close()


__all__ = ["main"]
