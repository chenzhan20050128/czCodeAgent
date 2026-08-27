"""Local tools owned by the mca runtime."""

from __future__ import annotations

import os
from pathlib import Path

from ..approval import ApprovalRequest
from .filesystem import FileSystemTools
from .registry import (
    SideEffect,
    ToolRegistry,
    ToolResult,
    ToolSpec,
    ToolValidationError,
    UnknownToolError,
    truncate_output,
)
from .search import SearchTools
from .shell import ShellRunner


def create_tool_registry(workspace: str | os.PathLike[str]) -> ToolRegistry:
    """Build the fixed six-tool contract for one workspace.

    Stateful handlers are bound to the canonical workspace at construction.
    """

    filesystem = FileSystemTools(Path(workspace))
    search = SearchTools(Path(workspace))
    shell = ShellRunner(Path(workspace))
    specs = [
        ToolSpec(
            name="read_file",
            description=(
                "Read a UTF-8 text file inside the workspace with 1-based "
                "line pagination and line-numbered output."
            ),
            schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "offset": {"type": "integer", "minimum": 1},
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": filesystem.max_read_lines,
                    },
                },
                "required": ["path"],
                "additionalProperties": False,
            },
            handler=filesystem.read_file,
            side_effect=SideEffect.NONE,
        ),
        ToolSpec(
            name="list_dir",
            description=(
                "List a workspace directory in deterministic name order; "
                "directories end in / and symbolic links in @."
            ),
            schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": filesystem.max_list_entries,
                    },
                },
                "required": [],
                "additionalProperties": False,
            },
            handler=filesystem.list_dir,
            side_effect=SideEffect.NONE,
        ),
        ToolSpec(
            name="grep",
            description=(
                "Search workspace files using ripgrep regular expressions and "
                "an optional file glob; ripgrep must be installed."
            ),
            schema={
                "type": "object",
                "properties": {
                    "pattern": {"type": "string"},
                    "path": {"type": "string"},
                    "glob": {"type": "string"},
                },
                "required": ["pattern"],
                "additionalProperties": False,
            },
            handler=search.grep,
            side_effect=SideEffect.NONE,
        ),
        ToolSpec(
            name="write_file",
            description=(
                "Prepare a complete UTF-8 file replacement inside the "
                "workspace for approval and conflict-checked atomic execution."
            ),
            schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
                "additionalProperties": False,
            },
            prepare_handler=filesystem.prepare_write_file,
            side_effect=SideEffect.WORKSPACE_WRITE,
            approval_renderer=lambda prepared: ApprovalRequest.for_file(
                "write_file", prepared
            ).render(),
        ),
        ToolSpec(
            name="edit_file",
            description=(
                "Prepare replacement of exactly one old_text occurrence in a "
                "UTF-8 workspace file for approval and atomic execution."
            ),
            schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "old_text": {"type": "string"},
                    "new_text": {"type": "string"},
                },
                "required": ["path", "old_text", "new_text"],
                "additionalProperties": False,
            },
            prepare_handler=filesystem.prepare_edit_file,
            side_effect=SideEffect.WORKSPACE_WRITE,
            approval_renderer=lambda prepared: ApprovalRequest.for_file(
                "edit_file", prepared
            ).render(),
        ),
        ToolSpec(
            name="bash",
            description=(
                "Execute an approved foreground command with /bin/sh -lc in "
                "the workspace, bounded output, and a timeout."
            ),
            schema={
                "type": "object",
                "properties": {
                    "command": {"type": "string", "minLength": 1},
                    "timeout_seconds": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 600,
                    },
                },
                "required": ["command"],
                "additionalProperties": False,
            },
            prepare_handler=shell.prepare,
            side_effect=SideEffect.SHELL,
            approval_renderer=lambda prepared: ApprovalRequest.for_shell(
                command=prepared.command, cwd=prepared.cwd
            ).render(),
        ),
    ]
    return ToolRegistry(specs)

__all__ = [
    "SideEffect",
    "ToolRegistry",
    "ToolResult",
    "ToolSpec",
    "ToolValidationError",
    "UnknownToolError",
    "create_tool_registry",
    "truncate_output",
]
