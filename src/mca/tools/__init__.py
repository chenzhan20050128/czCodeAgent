"""Local tools owned by the mca runtime."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from .registry import (
    SideEffect,
    ToolRegistry,
    ToolResult,
    ToolSpec,
    ToolValidationError,
    UnknownToolError,
    truncate_output,
)
from .filesystem import FileSystemTools
from .search import SearchTools


def create_tool_registry(workspace: str | os.PathLike[str]) -> ToolRegistry:
    """Build the fixed six-tool contract for one workspace.

    Shell preparation intentionally remains a fail-fast placeholder until the
    approval and process lifecycle are implemented in Task 6.
    """

    filesystem = FileSystemTools(Path(workspace))
    search = SearchTools(Path(workspace))
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
        ),
        ToolSpec(
            name="bash",
            description=(
                "Prepare a command for approved execution by the Task 6 shell "
                "runner in the workspace."
            ),
            schema={
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                    "timeout_seconds": {
                        "type": "number",
                        "minimum": 0.1,
                        "maximum": 600.0,
                    },
                },
                "required": ["command"],
                "additionalProperties": False,
            },
            prepare_handler=_prepare_bash_placeholder,
            side_effect=SideEffect.SHELL,
        ),
    ]
    return ToolRegistry(specs)


def _prepare_bash_placeholder(arguments: dict[str, Any]) -> object:
    del arguments
    raise NotImplementedError("bash execution is implemented in Task 6")

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
