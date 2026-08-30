"""Local tools owned by the mca runtime."""

from __future__ import annotations

import os
from pathlib import Path

from ..approval import ApprovalRequest
from .filesystem import FileSystemTools
from .plan import prepare_exit_plan_mode
from .registry import (
    ExecutionMode,
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

    from ..code_mode import prepare_code_program
    from ..code_sdk import render_python_sdk

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
                    "path": {"type": "string", "minLength": 1},
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
            is_concurrency_safe=lambda arguments: True,
        ),
        ToolSpec(
            name="list_dir",
            description=(
                "List a workspace directory in deterministic name order; "
                "directories end in / and symbolic links in @. Omit path to "
                "list the workspace root."
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
            is_concurrency_safe=lambda arguments: True,
        ),
        ToolSpec(
            name="grep",
            description=(
                "Search workspace files using ripgrep regular expressions and "
                "an optional file glob; ripgrep must be installed. Omit path to "
                "search the workspace root."
            ),
            schema={
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "minLength": 1},
                    "path": {"type": "string"},
                    "glob": {"type": "string", "minLength": 1},
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
                "workspace for approval and conflict-checked atomic execution. "
                "Missing parent directories are created automatically, so do "
                "not call bash mkdir first."
            ),
            schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "minLength": 1},
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
                "Prepare replacement of exactly one old_text occurrence in an "
                "existing UTF-8 workspace file for approval and atomic "
                "execution; use write_file to create a new file."
            ),
            schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "minLength": 1},
                    "old_text": {"type": "string", "minLength": 1},
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
        ToolSpec(
            name="exit_plan_mode",
            description=(
                "Use only in plan mode. Present your complete plan as markdown "
                "starting with a # heading and request approval to leave plan "
                "mode. Approval lets you carry out the plan from the next step; "
                "denial keeps plan mode on with the user's feedback."
            ),
            schema={
                "type": "object",
                "properties": {"plan": {"type": "string", "minLength": 1}},
                "required": ["plan"],
                "additionalProperties": False,
            },
            prepare_handler=prepare_exit_plan_mode,
            side_effect=SideEffect.PLAN_EXIT,
            approval_renderer=lambda prepared: (
                "Tool: exit_plan_mode\nApprove this plan and leave plan mode?\n"
                "Plan:\n" + prepared.plan
            ),
        ),
    ]
    base_registry = ToolRegistry(specs, workspace=Path(workspace))
    specs.append(ToolSpec(
        name="run_code",
        description=(
            "Execute a Constrained Python program that composes MCA tools. "
            "Only the program result and summary return to the model.\n\n"
            + render_python_sdk(base_registry)
        ),
        schema={
            "type": "object",
            "properties": {
                "description": {"type": "string", "minLength": 1},
                "code": {"type": "string", "minLength": 1},
            },
            "required": ["description", "code"],
            "additionalProperties": False,
        },
        prepare_handler=prepare_code_program,
        side_effect=SideEffect.NONE,
    ))
    return ToolRegistry(specs, workspace=Path(workspace))

__all__ = [
    "ExecutionMode",
    "SideEffect",
    "ToolRegistry",
    "ToolResult",
    "ToolSpec",
    "ToolValidationError",
    "UnknownToolError",
    "create_tool_registry",
    "truncate_output",
]
