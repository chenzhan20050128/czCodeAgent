"""Small, dependency-free terminal presentation and multiline editing helpers."""

from __future__ import annotations

import os
import sys
import termios
import tty


_RESET = "\x1b[0m"
# Deliberately muted 256-color palette. These are semantic roles, not raw
# red/yellow/green status lights, so the terminal stays readable on dark themes.
_PALETTE = {
    "info": "38;5;110",       # steel blue
    "workspace": "38;5;67",   # deep slate blue
    "model": "38;5;141",      # muted indigo
    "tool": "38;5;104",       # dusk violet
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
        while True:
            raw = os.read(descriptor, 1)
            if not raw:
                raise EOFError
            character = raw.decode("utf-8", errors="replace")
            if escape:
                escape += character
                if is_ctrl_enter_sequence(escape):
                    _write(output_stream, "\r\n")
                    return buffer.value.strip()
                candidates = ("\x1b[13;5u", "\x1b[27;5;13~", "\x1b[A", "\x1b[B", "\x1b[C", "\x1b[D")
                if escape in {"\x1b[A", "\x1b[B", "\x1b[C", "\x1b[D"}:
                    escape = ""
                    continue
                if any(candidate.startswith(escape) for candidate in candidates):
                    continue
                buffer.insert(escape)
                _write(output_stream, escape)
                escape = ""
                continue
            if character == "\x1b":
                escape = character
                continue
            if character == "\x13":
                _write(output_stream, "\r\n")
                return buffer.value.strip()
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
                continue
            if character in {"\r", "\n"}:
                buffer.newline()
                _write(output_stream, "\r\n" + continuation)
                continue
            if ord(character) >= 32:
                buffer.insert(character)
                _write(output_stream, character)
    finally:
        # Restore the terminal's ordinary keyboard protocol before returning
        # control to prompts, shells, or approval input.
        _write(output_stream, "\x1b[>4;0m\x1b[<u")
        termios.tcsetattr(descriptor, termios.TCSADRAIN, previous)


def _write(stream: object, value: str) -> None:
    stream.write(value)
    stream.flush()


__all__ = [
    "MultiLineBuffer",
    "TerminalInputError",
    "TerminalTheme",
    "is_ctrl_enter_sequence",
    "read_multiline_prompt",
]
