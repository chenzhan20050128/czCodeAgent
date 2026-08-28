"""Explicit, fail-closed approval for local side effects."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING
import unicodedata

if TYPE_CHECKING:
    from .tools.filesystem import PreparedFileChange


class ApprovalDecision(str, Enum):
    """A one-call or session-scoped user authorization decision."""

    ALLOW_ONCE = "allow_once"
    ALLOW_SESSION = "allow_session"
    DENY = "deny"


class ApprovalInterrupted(KeyboardInterrupt):
    """The operator cancelled approval; callers must stop the remaining batch."""


@dataclass(frozen=True)
class ApprovalRequest:
    """An exact human-readable description of one proposed side effect."""

    tool_name: str
    target: str
    kind: str
    cwd: str | None = None
    diff: str | None = None
    before_hash: str | None = None

    @classmethod
    def for_file(
        cls, tool_name: str, prepared: PreparedFileChange
    ) -> ApprovalRequest:
        return cls(
            tool_name=tool_name,
            target=str(prepared.canonical_path),
            kind="file",
            diff=prepared.diff,
            before_hash=prepared.before_hash,
        )

    @classmethod
    def for_shell(
        cls, *, command: str, cwd: str | Path
    ) -> ApprovalRequest:
        return cls(
            tool_name="bash",
            target=command,
            kind="shell",
            cwd=str(Path(cwd).resolve()),
        )

    def render(self) -> str:
        if self.kind == "file":
            before = self.before_hash if self.before_hash is not None else "<absent>"
            return (
                f"Tool: {_escape_terminal_text(self.tool_name)}\n"
                f"Path: {_escape_terminal_text(self.target)}\n"
                f"Before SHA-256: {before}\n"
                "Diff:\n"
                f"{_escape_terminal_text(self.diff or '', preserve_newlines=True)}"
            )
        if self.kind == "shell":
            return (
                "Tool: bash\n"
                "Shell: /bin/sh -lc\n"
                f"Cwd: {_escape_terminal_text(self.cwd or '')}\n"
                "Command:\n"
                f"{_escape_terminal_text(self.target)}\n"
                "Warning: shell commands may start descendant processes; "
                "MCA does not manage background jobs after command completion.\n"
            )
        if self.kind == "rendered":
            return self.target
        raise ValueError(f"unknown approval request kind: {self.kind}")


def _escape_terminal_text(value: str, *, preserve_newlines: bool = False) -> str:
    """Make untrusted text inert while retaining readable diff line structure."""

    rendered: list[str] = []
    for character in value:
        codepoint = ord(character)
        if character == "\n" and preserve_newlines:
            rendered.append(character)
        elif character == "\n":
            rendered.append(r"\n")
        elif character == "\r":
            rendered.append(r"\r")
        elif character == "\t":
            rendered.append(r"\t")
        elif character == "\b":
            rendered.append(r"\b")
        elif codepoint <= 0xFF and (
            codepoint < 0x20 or 0x7F <= codepoint <= 0x9F
        ):
            rendered.append(f"\\x{codepoint:02x}")
        elif unicodedata.category(character) == "Cf":
            rendered.append(f"\\u{codepoint:04x}")
        else:
            rendered.append(character)
    return "".join(rendered)


class InteractiveApprover:
    """Ask once per request, with an explicit current-session always mode."""

    def __init__(
        self,
        *,
        yolo: bool = False,
        input_fn: Callable[[str], str] = input,
        output_fn: Callable[[str], object] = print,
    ) -> None:
        self._yolo = bool(yolo)
        self._input = input_fn
        self._output = output_fn
        self._session_always = False

    def decide(self, request: ApprovalRequest) -> ApprovalDecision:
        if self._yolo:
            return ApprovalDecision.ALLOW_ONCE
        if self._session_always:
            return ApprovalDecision.ALLOW_SESSION
        self._output(request.render())
        try:
            answer = self._input("Allow once? [y/N/always] ")
        except EOFError:
            return ApprovalDecision.DENY
        except KeyboardInterrupt:
            raise ApprovalInterrupted from None
        if answer.strip().lower() in {"y", "yes"}:
            return ApprovalDecision.ALLOW_ONCE
        if answer.strip().lower() == "always":
            self._session_always = True
            return ApprovalDecision.ALLOW_SESSION
        return ApprovalDecision.DENY

    def reset_session_approval(self) -> None:
        """Return this interactive approver to one-time confirmation mode."""

        self._session_always = False


__all__ = [
    "ApprovalDecision",
    "ApprovalInterrupted",
    "ApprovalRequest",
    "InteractiveApprover",
]
