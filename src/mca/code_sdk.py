"""Stable Python SDK projection for Code Mode-visible tools."""

from __future__ import annotations

from typing import Any

from .tools.registry import ToolRegistry


_EXCLUDED = frozenset({"run_code", "exit_plan_mode"})


def _annotation(schema: dict[str, Any]) -> str:
    variants = schema.get("anyOf")
    if isinstance(variants, list):
        return " | ".join(_annotation(item) for item in variants)
    return {
        "string": "str",
        "integer": "int",
        "number": "int | float",
        "boolean": "bool",
        "array": "list[JsonValue]",
        "object": "dict[str, JsonValue]",
    }.get(str(schema.get("type")), "JsonValue")


def render_python_sdk(registry: ToolRegistry) -> str:
    """Render a deterministic model-facing declaration and usage guide."""

    lines = [
        "Constrained Python Code Mode SDK",
        "Tool calls create lazy ToolNode values. Use await node for dependencies,",
        "await gather(a, b) for declared parallel work, and after=[a, b] to",
        "prevent downstream execution when an upstream node fails. Failures use",
        "ToolCallError; blocked descendants report UPSTREAM_FAILED.",
        "Every awaited tool returns a ToolResult object; read its output field",
        "instead of treating the whole object as text. End with an explicit",
        "return of a JSON-compatible value. A final bare expression is discarded.",
        "Imports and json.dumps/json.dump are unavailable and unnecessary.",
        "read_file output is line-numbered for display. For edit_file, use a",
        "known unique source substring as old_text; do not submit line prefixes.",
        "",
        "JsonValue = None | bool | int | float | str | list[JsonValue] | dict[str, JsonValue]",
        'ToolResult = {"status": str, "output": str, "exit_code": int | None, "truncated": bool, "metadata": dict[str, JsonValue]}',
        "",
        "class ToolNode:",
        "    node_id: str",
        "    def after(self, *dependencies: ToolNode) -> ToolNode: ...",
        "    def __await__(self): ...",
        "",
        "class Tools:",
    ]
    for schema in registry.provider_schemas():
        function = schema["function"]
        name = function["name"]
        if name in _EXCLUDED:
            continue
        parameters = function["parameters"]
        properties = parameters.get("properties", {})
        required = set(parameters.get("required", []))
        fields = ", ".join(
            f"{key}: {_annotation(value)}{'' if key in required else ' | None'}"
            for key, value in sorted(properties.items())
        )
        lines.extend(
            [
                f"    # {function['description']}",
                f"    def {name}(self, args: dict[str, JsonValue], *, after: list[ToolNode] = []) -> ToolNode: ...  # {fields}",
            ]
        )
    lines.extend(
        [
            "",
            "tools: Tools",
        "async def gather(*nodes: ToolNode) -> list[ToolResult]: ...",
        "async def execute(*nodes: ToolNode) -> list[ToolResult]: ...",
        "",
        "Example (two independent reads, one explicit JSON-compatible return):",
        'readme = tools.read_file({"path": "README.md"})',
        'config = tools.read_file({"path": ".env.example"})',
        "results = await gather(readme, config)",
        'readme_text = results[0]["output"]',
        'config_text = results[1]["output"]',
        'return {"readme_length": len(readme_text), "has_config": "MCA_" in config_text}',
        ]
    )
    return "\n".join(lines)


__all__ = ["render_python_sdk"]
