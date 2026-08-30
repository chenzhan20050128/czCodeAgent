"""Small, dependency-free terminal presentation and multiline editing helpers."""

from __future__ import annotations

import codecs
import os
import sys
import termios
import tty
import unicodedata

from .approval import _escape_terminal_text
from .code_graph import CodeGraphNodeView, CodeGraphView


_RESET = "\x1b[0m"
# Deliberately muted 256-color palette. These are semantic roles, not raw
# red/yellow/green status lights, so the terminal stays readable on dark themes.
_PALETTE = {
    "info": "38;5;110",       # steel blue
    "workspace": "38;5;67",   # deep slate blue
    "model": "38;5;141",      # muted indigo
    "tool": "38;5;130",       # deep burnt amber
    "approval": "38;5;137",   # muted amber-brown
    "success": "38;5;72",     # deep teal-green
    "failure": "38;5;167",    # dusty brick
    "muted": "38;5;245",      # stone gray
    "prompt": "38;5;117",     # pale blue
}


class TerminalTheme:
    """Render a restrained ANSI theme, with a safe plain-text fallback."""

    def __init__(self, *, enabled: bool) -> None:
        self.enabled = bool(enabled)

    @classmethod
    def auto(cls, *, isatty: bool) -> TerminalTheme:
        disabled = bool(os.environ.get("NO_COLOR")) or os.environ.get("TERM") == "dumb"
        return cls(enabled=isatty and not disabled)

    def style(self, text: str, role: str) -> str:
        if not self.enabled:
            return text
        color = _PALETTE.get(role, _PALETTE["muted"])
        return f"\x1b[{color}m{text}{_RESET}"

    def label(self, text: str) -> str:
        return self.style(text, "info")


_GRAPH_STATUS = {
    "planned": ("○", "PLANNED"),
    "started": ("▶", "RUNNING"),
    "succeeded": ("✓", "SUCCEEDED"),
    "user_confirmed_success": ("✓", "CONFIRMED"),
    "failed": ("✗", "FAILED"),
    "denied": ("⊘", "DENIED"),
    "invalid_arguments": ("✗", "INVALID"),
    "unknown_tool": ("✗", "UNKNOWN_TOOL"),
    "conflict": ("✗", "CONFLICT"),
    "timed_out": ("◷", "TIMED_OUT"),
    "interrupted": ("!", "INTERRUPTED"),
    "cancelled": ("⊘", "CANCELLED"),
    "not_executed": ("⊘", "NOT_EXECUTED"),
    "upstream_failed": ("⊘", "UPSTREAM_FAILED"),
    "outcome_unknown": ("?", "OUTCOME_UNKNOWN"),
    "abandoned": ("⊘", "ABANDONED"),
    "batch_limit_exceeded": ("⊘", "BATCH_LIMIT"),
    "user_confirmed_failure": ("✗", "CONFIRMED_FAILED"),
}
_GRAPH_FAILURE_STATUSES = frozenset(
    {
        "failed",
        "denied",
        "invalid_arguments",
        "unknown_tool",
        "conflict",
        "timed_out",
        "interrupted",
        "cancelled",
        "upstream_failed",
        "outcome_unknown",
        "abandoned",
        "batch_limit_exceeded",
        "user_confirmed_failure",
    }
)


def render_code_graph_plain(
    graph: CodeGraphView, *, width: int = 100, expanded: bool = True
) -> str:
    """Render a stable complete DAG without terminal control sequences."""

    if not isinstance(graph, CodeGraphView):
        raise TypeError("graph must be a CodeGraphView")
    if type(width) is not int or width < 20:
        raise ValueError("width must be at least 20")
    lines = [
        f"╭─ run_code: {_graph_text(graph.description)} {graph.status.upper()}",
    ]
    for node in graph.nodes:
        lines.append(_render_graph_node(node))
        if expanded and node.result and node.status in _GRAPH_FAILURE_STATUSES:
            lines.append(f"│     {_one_line(node.result)}")
        if node.blocked_by_ordinals:
            blockers = ", ".join(f"#{value}" for value in node.blocked_by_ordinals)
            lines.append(f"│     blocked by {blockers}")
    if expanded:
        for node in graph.nodes:
            for dependent in node.dependent_ordinals:
                lines.append(f"│  #{node.ordinal} ──▶ #{dependent}")
    if graph.shell_mutation_warning:
        lines.append(
            "│  warning: parallel bash + file mutation may contend for workspace resources"
        )
    lines.append(f"╰─ {_render_graph_summary(graph)}")
    return "\n".join(_fit_graph_line(line, width) for line in lines)


def render_code_graph_ansi(
    graph: CodeGraphView, *, width: int = 100, expanded: bool = True
) -> str:
    """Render the same graph content with restrained semantic coloring."""

    theme = TerminalTheme(enabled=True)
    plain_lines = render_code_graph_plain(
        graph, width=width, expanded=expanded
    ).splitlines()
    styled: list[str] = []
    for line in plain_lines:
        if "CURRENT" in line:
            role = "prompt"
        elif any(label in line for label in ("FAILED", "CONFLICT", "DENIED", "UNKNOWN", "INVALID")):
            role = "failure"
        elif "SUCCEEDED" in line or "✓" in line:
            role = "success"
        elif "warning:" in line or "UPSTREAM" in line:
            role = "approval"
        elif line.startswith(("╭", "╰")):
            role = "tool"
        else:
            role = "muted"
        styled.append(theme.style(line, role))
    return "\n".join(styled)


def _render_graph_node(node: CodeGraphNodeView) -> str:
    symbol, label = _GRAPH_STATUS.get(node.status, ("?", node.status.upper()))
    dependencies = (
        " after " + ",".join(f"#{value}" for value in node.dependency_ordinals)
        if node.dependency_ordinals
        else ""
    )
    target = f"  {_graph_text(node.target)}" if node.target else ""
    elapsed = f"  {_format_elapsed(node.elapsed_ms)}" if node.elapsed_ms is not None else ""
    current = "  CURRENT" if node.is_current else ""
    approval = (
        f"  {node.approval.upper()}" if node.approval is not None else ""
    )
    return (
        f"│  #{node.ordinal} {symbol} {label}{approval}{current}"
        f"  {_graph_text(node.name)}{target}{dependencies}{elapsed}"
    )


def _render_graph_summary(graph: CodeGraphView) -> str:
    labels = (
        "succeeded", "failed", "denied", "conflict", "timed_out",
        "interrupted", "upstream_failed", "not_executed",
        "outcome_unknown",
    )
    parts = [
        f"{graph.summary.get(label, 0)} {label}"
        for label in labels
        if graph.summary.get(label, 0)
    ]
    if not parts:
        parts.append(f"{graph.summary.get('planned', 0)} planned")
    if graph.elapsed_ms is not None:
        parts.append(f"wall {_format_elapsed(graph.elapsed_ms)}")
    return " · ".join(parts)


def _format_elapsed(milliseconds: int) -> str:
    if milliseconds < 1000:
        return f"{milliseconds} ms"
    return f"{milliseconds / 1000:.1f} s"


def _one_line(value: str) -> str:
    return _graph_text(value)


def _graph_text(value: str) -> str:
    return _escape_terminal_text(value)


def _fit_graph_line(line: str, width: int) -> str:
    if _display_width(line) <= width:
        return line
    budget = max(0, width - 1)
    rendered: list[str] = []
    used = 0
    for character in line:
        cell_width = _character_width(character)
        if used + cell_width > budget:
            break
        rendered.append(character)
        used += cell_width
    return "".join(rendered) + "…"


def _display_width(value: str) -> int:
    return sum(_character_width(character) for character in value)


def _character_width(character: str) -> int:
    if unicodedata.combining(character):
        return 0
    return 2 if unicodedata.east_asian_width(character) in {"W", "F"} else 1


class MultiLineBuffer:
    """A minimal append/backspace editor for prompt text in raw terminal mode."""

    def __init__(self) -> None:
        self._characters: list[str] = []

    @property
    def value(self) -> str:
        return "".join(self._characters)

    def insert(self, character: str) -> None:
        if not isinstance(character, str) or len(character) != 1:
            raise ValueError("character must be exactly one character")
        self._characters.append(character)

    def newline(self) -> None:
        self._characters.append("\n")

    def backspace(self) -> None:
        if self._characters:
            self._characters.pop()


def is_ctrl_enter_sequence(value: str) -> bool:
    """Return true for the portable Ctrl+Enter extensions we explicitly support."""

    return value in {"\x1b[13;5u", "\x1b[27;5;13~"}


class TerminalInputError(RuntimeError):
    """The terminal cannot enter the requested interactive input mode."""


def read_multiline_prompt(
    *,
    input_stream: object = sys.stdin,
    output_stream: object = sys.stdout,
    prompt: str = "mca> ",
    continuation: str = "...  ",
) -> str:
    """Read in raw mode; Enter inserts a line break, Ctrl+Enter submits.

    Ctrl+Enter is recognized as CSI-u (``ESC [ 13 ; 5 u``) and xterm's
    modifyOtherKeys sequence (``ESC [ 27 ; 5 ; 13 ~``). Ctrl+S is retained as
    an explicit cross-terminal fallback because many terminals encode Ctrl+Enter
    as an ordinary CR, indistinguishable from Enter.
    """

    if not hasattr(input_stream, "isatty") or not input_stream.isatty():
        raise TerminalInputError("multiline input requires an interactive terminal")
    if not hasattr(input_stream, "fileno"):
        raise TerminalInputError("multiline input stream has no file descriptor")
    descriptor = input_stream.fileno()
    try:
        previous = termios.tcgetattr(descriptor)
    except termios.error as error:
        raise TerminalInputError("could not configure terminal input") from error

    buffer = MultiLineBuffer()
    _write(output_stream, prompt)
    try:
        tty.setraw(descriptor)
        # Ask compatible terminals to report modified key combinations in a
        # distinguishable form. Unsupported terminals ignore this request; the
        # documented Ctrl+S fallback remains available.
        _write(output_stream, "\x1b[>4;2m\x1b[>1u")
        escape = ""
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        while True:
            raw = os.read(descriptor, 1)
            if not raw:
                raise EOFError
            decoded = decoder.decode(raw, final=False)
            if not decoded:
                continue
            for character in decoded:
                submitted = _consume_character(
                    character, buffer, output_stream, prompt, continuation, escape
                )
                escape = submitted.escape
                if submitted.value is not None:
                    return submitted.value
    finally:
        # Restore the terminal's ordinary keyboard protocol before returning
        # control to prompts, shells, or approval input.
        _write(output_stream, "\x1b[>4;0m\x1b[<u")
        termios.tcsetattr(descriptor, termios.TCSADRAIN, previous)


def _write(stream: object, value: str) -> None:
    stream.write(value)
    stream.flush()


class _EditResult:
    def __init__(self, escape: str, value: str | None = None) -> None:
        self.escape = escape
        self.value = value


def _consume_character(
    character: str,
    buffer: MultiLineBuffer,
    output_stream: object,
    prompt: str,
    continuation: str,
    escape: str,
) -> _EditResult:
    """Consume one decoded Unicode character from the raw terminal."""

    if escape:
        escape += character
        if is_ctrl_enter_sequence(escape):
            _write(output_stream, "\r\n")
            return _EditResult("", buffer.value.strip())
        candidates = ("\x1b[13;5u", "\x1b[27;5;13~", "\x1b[A", "\x1b[B", "\x1b[C", "\x1b[D")
        if escape in {"\x1b[A", "\x1b[B", "\x1b[C", "\x1b[D"}:
            return _EditResult("")
        if any(candidate.startswith(escape) for candidate in candidates):
            return _EditResult(escape)
        buffer.insert(escape)
        _write(output_stream, escape)
        return _EditResult("")
    if character == "\x1b":
        return _EditResult(character)
    if character == "\x13":
        _write(output_stream, "\r\n")
        return _EditResult("", buffer.value.strip())
    if character == "\x03":
        raise KeyboardInterrupt
    if character == "\x04" and not buffer.value:
        raise EOFError
    if character in {"\x7f", "\x08"}:
        if buffer.value:
            was_newline = buffer.value.endswith("\n")
            buffer.backspace()
            if was_newline:
                _write(output_stream, "\r\x1b[2K" + prompt + buffer.value.rsplit("\n", 1)[-1])
            else:
                _write(output_stream, "\b \b")
        return _EditResult("")
    if character in {"\r", "\n"}:
        buffer.newline()
        _write(output_stream, "\r\n" + continuation)
        return _EditResult("")
    if ord(character) >= 32:
        buffer.insert(character)
        _write(output_stream, character)
    return _EditResult("")


__all__ = [
    "render_code_graph_ansi",
    "render_code_graph_plain",
    "MultiLineBuffer",
    "TerminalInputError",
    "TerminalTheme",
    "is_ctrl_enter_sequence",
    "read_multiline_prompt",
]
