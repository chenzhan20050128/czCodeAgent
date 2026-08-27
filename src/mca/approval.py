"""Explicit, fail-closed approval for local side effects."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .tools.filesystem import PreparedFileChange


class ApprovalDecision(str, Enum):
    """The complete approval vocabulary; authorization is never cached."""

    ALLOW_ONCE = "allow_once"
    DENY = "deny"


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
                f"Tool: {self.tool_name}\n"
                f"Path: {self.target}\n"
                f"Before SHA-256: {before}\n"
                "Diff:\n"
                f"{self.diff or ''}"
            )
        if self.kind == "shell":
            return (
                "Tool: bash\n"
                "Shell: /bin/sh -lc\n"
                f"Cwd: {self.cwd}\n"
                "Command:\n"
                f"{self.target}\n"
            )
        raise ValueError(f"unknown approval request kind: {self.kind}")


class InteractiveApprover:
    """Ask once per request and deny every non-explicit affirmative answer."""

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

    def decide(self, request: ApprovalRequest) -> ApprovalDecision:
        if self._yolo:
            return ApprovalDecision.ALLOW_ONCE
        self._output(request.render())
        try:
            answer = self._input("Allow once? [y/N] ")
        except (EOFError, KeyboardInterrupt):
            return ApprovalDecision.DENY
        if answer.strip().lower() in {"y", "yes"}:
            return ApprovalDecision.ALLOW_ONCE
        return ApprovalDecision.DENY


__all__ = ["ApprovalDecision", "ApprovalRequest", "InteractiveApprover"]
