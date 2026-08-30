"""Explicit tool specifications, argument validation, and bounded results."""

from __future__ import annotations

import json
import math
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


TRUNCATION_MARKER = "... [truncated] ..."
DEFAULT_MAX_OUTPUT_BYTES = 64 * 1024
DEFAULT_MAX_OUTPUT_LINES = 1_000
_SUPPORTED_TYPES = frozenset(
    {"object", "string", "integer", "number", "boolean", "array"}
)

ToolHandler = Callable[[dict[str, Any]], object]


class ToolValidationError(ValueError):
    """Raised when provider arguments do not satisfy a tool's schema."""


class UnknownToolError(LookupError):
    """Raised when a provider asks for a tool that is not registered."""


class SideEffect(str, Enum):
    """The externally visible effect class of a local tool."""

    NONE = "none"
    WORKSPACE_WRITE = "workspace_write"
    SHELL = "shell"
    PLAN_EXIT = "plan_exit"


class ExecutionMode(str, Enum):
    """Host scheduling mode for one validated tool call."""

    PARALLEL = "parallel"
    EXCLUSIVE = "exclusive"


@dataclass(frozen=True)
class ToolSpec:
    """One handwritten provider contract and its local entry point."""

    name: str
    description: str
    schema: Mapping[str, Any]
    handler: ToolHandler | None = None
    prepare_handler: ToolHandler | None = None
    side_effect: SideEffect | bool = SideEffect.NONE
    approval_renderer: Callable[[object], str] | None = None
    is_concurrency_safe: Callable[[dict[str, Any]], bool] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("tool name must be a non-empty string")
        if not isinstance(self.description, str) or not self.description:
            raise ValueError("tool description must be a non-empty string")
        if (self.handler is None) == (self.prepare_handler is None):
            raise ValueError("tool must define exactly one handler or prepare handler")
        if not isinstance(self.side_effect, (SideEffect, bool)):
            raise ValueError("side_effect must be a SideEffect or boolean")
        if self.is_concurrency_safe is not None and not callable(
            self.is_concurrency_safe
        ):
            raise ValueError("is_concurrency_safe must be callable or None")
        _check_schema(self.schema, path="schema")
        if self.schema.get("type") != "object":
            raise ValueError("tool schema must have type object")

    def provider_schema(self) -> dict[str, object]:
        """Return the OpenAI-compatible function-tool declaration."""

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.schema,
            },
        }


@dataclass(frozen=True)
class ToolResult:
    """The only model-visible record retained from one tool execution."""

    title: str
    output: str
    metadata: Mapping[str, object] = field(default_factory=dict)
    status: str = "succeeded"

    def __post_init__(self) -> None:
        if not isinstance(self.title, str) or not self.title:
            raise ValueError("tool result title must be a non-empty string")
        if not isinstance(self.output, str):
            raise ValueError("tool result output must be a string")
        if not isinstance(self.metadata, Mapping):
            raise ValueError("tool result metadata must be an object")
        if not isinstance(self.status, str) or not self.status:
            raise ValueError("tool result status must be a non-empty string")

    @classmethod
    def bounded(
        cls,
        *,
        title: str,
        output: str,
        status: str = "succeeded",
        metadata: Mapping[str, object] | None = None,
        max_bytes: int = DEFAULT_MAX_OUTPUT_BYTES,
        max_lines: int = DEFAULT_MAX_OUTPUT_LINES,
    ) -> ToolResult:
        bounded_output, truncated = truncate_output(
            output, max_bytes=max_bytes, max_lines=max_lines
        )
        result_metadata = dict(metadata or {})
        result_metadata["truncated"] = bool(
            result_metadata.get("truncated", False)
        ) or truncated
        return cls(
            title=title,
            output=bounded_output,
            metadata=result_metadata,
            status=status,
        )


class ToolRegistry:
    """A fixed, name-addressable set of explicit tool specifications."""

    def __init__(
        self,
        specs: Sequence[ToolSpec] = (),
        *,
        workspace: str | os.PathLike[str] | None = None,
    ) -> None:
        self.workspace = (
            Path(workspace).resolve(strict=True) if workspace is not None else None
        )
        if self.workspace is not None and not self.workspace.is_dir():
            raise ValueError("registry workspace must be a directory")
        self._specs: dict[str, ToolSpec] = {}
        for spec in specs:
            if spec.name in self._specs:
                raise ValueError(f"duplicate tool: {spec.name}")
            self._specs[spec.name] = spec

    def resolve(self, name: str) -> ToolSpec:
        try:
            return self._specs[name]
        except (KeyError, TypeError):
            raise UnknownToolError(f"unknown tool: {name}") from None

    def provider_schemas(self) -> list[dict[str, object]]:
        return [spec.provider_schema() for spec in self._specs.values()]

    def parse_and_validate(self, name: str, raw_arguments: str) -> dict[str, Any]:
        spec = self.resolve(name)
        if not isinstance(raw_arguments, str):
            raise ToolValidationError("arguments must be a JSON string")
        try:
            arguments = json.loads(
                raw_arguments,
                parse_constant=lambda value: (_raise_invalid_constant(value)),
            )
        except (json.JSONDecodeError, ValueError):
            raise ToolValidationError("arguments must be valid JSON") from None
        if not isinstance(arguments, dict):
            raise ToolValidationError("arguments must be an object")
        validate_arguments(spec.schema, arguments)
        return arguments

    def execution_mode(self, name: str, raw_arguments: str) -> ExecutionMode:
        """Classify a pending call, failing closed on every uncertainty."""

        try:
            spec = self.resolve(name)
            if spec.side_effect not in {SideEffect.NONE, False}:
                return ExecutionMode.EXCLUSIVE
            classifier = spec.is_concurrency_safe
            if classifier is None:
                return ExecutionMode.EXCLUSIVE
            arguments = self.parse_and_validate(name, raw_arguments)
            safe = classifier(arguments)
        except Exception:
            return ExecutionMode.EXCLUSIVE
        return ExecutionMode.PARALLEL if safe is True else ExecutionMode.EXCLUSIVE


def _raise_invalid_constant(value: str) -> object:
    raise ValueError(f"invalid JSON constant: {value}")


def validate_arguments(schema: Mapping[str, Any], arguments: Mapping[str, Any]) -> None:
    """Validate arguments using the finite JSON Schema subset we advertise."""

    _validate_value(schema, arguments, path="arguments")


def _check_schema(schema: object, *, path: str) -> None:
    if not isinstance(schema, Mapping):
        raise ValueError(f"{path} must be an object")
    if "anyOf" in schema:
        variants = schema["anyOf"]
        if not isinstance(variants, list) or not variants:
            raise ValueError(f"{path}.anyOf must be a non-empty array")
        for index, variant in enumerate(variants):
            _check_schema(variant, path=f"{path}.anyOf[{index}]")
        return
    schema_type = schema.get("type")
    if schema_type not in _SUPPORTED_TYPES:
        raise ValueError(f"{path}.type is unsupported: {schema_type!r}")
    enum = schema.get("enum")
    if enum is not None and (not isinstance(enum, list) or not enum):
        raise ValueError(f"{path}.enum must be a non-empty array")
    if "minLength" in schema and (
        schema_type != "string"
        or type(schema["minLength"]) is not int
        or schema["minLength"] < 0
    ):
        raise ValueError(f"{path}.minLength must be a non-negative integer")
    for bound in ("minimum", "maximum"):
        if bound in schema and (
            isinstance(schema[bound], bool)
            or not isinstance(schema[bound], (int, float))
            or not math.isfinite(schema[bound])
        ):
            raise ValueError(f"{path}.{bound} must be a finite number")
    if schema_type == "object":
        properties = schema.get("properties")
        required = schema.get("required", [])
        if not isinstance(properties, Mapping):
            raise ValueError(f"{path}.properties must be an object")
        if not isinstance(required, list) or any(
            not isinstance(item, str) for item in required
        ):
            raise ValueError(f"{path}.required must be an array of strings")
        if len(set(required)) != len(required):
            raise ValueError(f"{path}.required must not contain duplicates")
        if any(item not in properties for item in required):
            raise ValueError(f"{path}.required contains an unknown property")
        if schema.get("additionalProperties") is not False:
            raise ValueError(f"{path}.additionalProperties must be false")
        for name, child in properties.items():
            if not isinstance(name, str):
                raise ValueError(f"{path}.properties keys must be strings")
            _check_schema(child, path=f"{path}.properties.{name}")
    if schema_type == "array":
        if "items" not in schema:
            raise ValueError(f"{path}.items is required")
        _check_schema(schema["items"], path=f"{path}.items")


def _validate_value(schema: Mapping[str, Any], value: object, *, path: str) -> None:
    variants = schema.get("anyOf")
    if variants is not None:
        for variant in variants:
            try:
                _validate_value(variant, value, path=path)
                return
            except ToolValidationError:
                pass
        raise ToolValidationError(f"{_display_path(path)} does not match any allowed schema")

    schema_type = schema["type"]
    label = _display_path(path)
    valid = {
        "object": lambda item: isinstance(item, dict),
        "string": lambda item: isinstance(item, str),
        "integer": lambda item: type(item) is int,
        "number": lambda item: (
            type(item) in (int, float) and math.isfinite(item)
        ),
        "boolean": lambda item: type(item) is bool,
        "array": lambda item: isinstance(item, list),
    }[schema_type]
    article = "an" if schema_type in {"object", "integer", "array"} else "a"
    if not valid(value):
        raise ToolValidationError(f"{label} must be {article} {schema_type}")

    if schema_type == "string" and len(value) < schema.get("minLength", 0):
        raise ToolValidationError(f"{label} must not be empty")

    enum = schema.get("enum")
    if enum is not None and not any(
        type(value) is type(candidate) and value == candidate for candidate in enum
    ):
        raise ToolValidationError(f"{label} must be one of {enum!r}")
    if "minimum" in schema and value < schema["minimum"]:
        raise ToolValidationError(f"{label} must be >= {schema['minimum']}")
    if "maximum" in schema and value > schema["maximum"]:
        raise ToolValidationError(f"{label} must be <= {schema['maximum']}")

    if schema_type == "object":
        properties = schema["properties"]
        for required in schema.get("required", []):
            if required not in value:
                raise ToolValidationError(f"missing required property: {required}")
        unknown = sorted(set(value) - set(properties))
        if unknown:
            raise ToolValidationError(f"unknown property: {unknown[0]}")
        for name, child_value in value.items():
            _validate_value(properties[name], child_value, path=f"{path}.{name}")
    elif schema_type == "array":
        for index, item in enumerate(value):
            _validate_value(schema["items"], item, path=f"{path}[{index}]")


def _display_path(path: str) -> str:
    return path.removeprefix("arguments.")


def truncate_output(
    output: str,
    *,
    max_bytes: int = DEFAULT_MAX_OUTPUT_BYTES,
    max_lines: int = DEFAULT_MAX_OUTPUT_LINES,
) -> tuple[str, bool]:
    """Bound UTF-8 output by bytes and lines while retaining both ends."""

    if not isinstance(output, str):
        raise TypeError("output must be a string")
    if type(max_bytes) is not int or max_bytes < 1:
        raise ValueError("max_bytes must be a positive integer")
    if type(max_lines) is not int or max_lines < 1:
        raise ValueError("max_lines must be a positive integer")
    lines = output.splitlines()
    line_count = len(lines) or (1 if output else 0)
    if len(output.encode("utf-8")) <= max_bytes and line_count <= max_lines:
        return output, False

    marker_bytes = TRUNCATION_MARKER.encode("utf-8")
    if max_lines == 1 or max_bytes <= len(marker_bytes) + 2:
        return _utf8_prefix(marker_bytes, max_bytes).decode("utf-8"), True

    source_lines = lines or [""]
    if max_lines == 2:
        prefix_source = source_lines[0]
        suffix_source = source_lines[-1]
        separator = f" {TRUNCATION_MARKER} "
    else:
        content_slots = min(len(source_lines), max_lines - 1)
        if len(source_lines) == 1:
            head_count = tail_count = 1
        else:
            head_count = max(1, (content_slots + 1) // 2)
            tail_count = max(1, content_slots - head_count)
        prefix_source = "\n".join(source_lines[:head_count])
        suffix_source = "\n".join(source_lines[-tail_count:])
        separator = f"\n{TRUNCATION_MARKER}\n"

    separator_bytes = separator.encode("utf-8")
    if len(separator_bytes) >= max_bytes:
        return _utf8_prefix(marker_bytes, max_bytes).decode("utf-8"), True
    available = max_bytes - len(separator_bytes)
    head_budget = available // 2
    tail_budget = available - head_budget
    prefix = _utf8_prefix(prefix_source.encode("utf-8"), head_budget)
    suffix = _utf8_suffix(suffix_source.encode("utf-8"), tail_budget)
    bounded = prefix.decode("utf-8") + separator + suffix.decode("utf-8")
    return bounded, True


def _utf8_prefix(data: bytes, limit: int) -> bytes:
    clipped = data[:limit]
    while clipped:
        try:
            clipped.decode("utf-8")
            return clipped
        except UnicodeDecodeError:
            clipped = clipped[:-1]
    return b""


def _utf8_suffix(data: bytes, limit: int) -> bytes:
    clipped = data[-limit:] if limit else b""
    while clipped:
        try:
            clipped.decode("utf-8")
            return clipped
        except UnicodeDecodeError:
            clipped = clipped[1:]
    return b""
