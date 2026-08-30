"""Isolated worker entry for constrained Python Code Mode programs."""

from __future__ import annotations

import asyncio
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import resource as _resource
except ImportError:  # pragma: no cover - non-POSIX fallback
    _resource = None

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mca.code_ast import (
    CodeValidationError,
    CollectionLimitError,
    EvaluationLimitError,
    Evaluator,
    GraphExecutionError,
    ToolCallError,
    ensure_json_value,
    validate_code,
)
from mca.code_protocol import decode_frame, encode_frame


class _ReturnStream:
    def __init__(self) -> None:
        self.logs: list[str] = []

    def print(self, *values: object) -> None:
        self.logs.append(" ".join(str(value) for value in values))


@dataclass
class ToolNode:
    client: _GraphClient
    node_id: str
    ordinal: int
    name: str
    arguments: dict[str, Any]
    dependencies: list[str] = field(default_factory=list)
    submitted: bool = False
    outcome: dict[str, Any] | None = None

    def after(self, *dependencies: ToolNode) -> ToolNode:
        if self.submitted:
            raise RuntimeError("NODE_ALREADY_SUBMITTED")
        for dependency in dependencies:
            if not isinstance(dependency, ToolNode) or dependency.client is not self.client:
                raise RuntimeError("FOREIGN_NODE_REFERENCE")
            if dependency.node_id == self.node_id:
                raise RuntimeError("CYCLIC_DEPENDENCY")
            if dependency.node_id not in self.dependencies:
                self.dependencies.append(dependency.node_id)
        return self

    def __await__(self):
        return self.client.execute((self,)).__await__()


class _Tools:
    def __init__(self, client: _GraphClient, names: tuple[str, ...]) -> None:
        self._client = client
        self._names = frozenset(names)

    def __getattr__(self, name: str):
        if name.startswith("_") or name not in self._names:
            raise AttributeError(name)

        def create(
            arguments: dict[str, Any], *, after: list[ToolNode] | tuple[ToolNode, ...] = ()
        ) -> ToolNode:
            return self._client.create(name, arguments, after)

        return create


class _GraphClient:
    def __init__(self, names: tuple[str, ...]) -> None:
        self.names = names
        self.nodes: dict[str, ToolNode] = {}
        self.next_ordinal = 1

    def create(
        self, name: str, arguments: dict[str, Any], after: list[ToolNode] | tuple[ToolNode, ...]
    ) -> ToolNode:
        if not isinstance(arguments, dict):
            raise TypeError("tool arguments must be an object")
        ensure_json_value(arguments)
        dependencies: list[str] = []
        for dependency in after:
            if not isinstance(dependency, ToolNode) or dependency.client is not self:
                raise RuntimeError("FOREIGN_NODE_REFERENCE")
            if dependency.node_id not in dependencies:
                dependencies.append(dependency.node_id)
        ordinal = self.next_ordinal
        self.next_ordinal += 1
        node_id = f"node-{ordinal}"
        node = ToolNode(self, node_id, ordinal, name, dict(arguments), dependencies)
        self.nodes[node_id] = node
        return node

    async def execute(self, targets: tuple[ToolNode, ...]) -> Any:
        for target in targets:
            if not isinstance(target, ToolNode) or target.client is not self:
                raise RuntimeError("FOREIGN_NODE_REFERENCE")
        pending = self._closure(targets)
        if pending:
            request = {
                "type": "execute_graph",
                "targets": [target.node_id for target in targets],
                "nodes": [
                    {
                        "node_id": node.node_id,
                        "ordinal": node.ordinal,
                        "name": node.name,
                        "arguments": node.arguments,
                        "dependencies": node.dependencies,
                    }
                    for node in pending
                ],
            }
            _send(request)
            response = _read()
            if response.get("type") != "graph_result":
                raise RuntimeError("invalid graph result frame")
            results = response.get("results")
            if not isinstance(results, dict):
                raise RuntimeError("graph result requires results")
            for node in pending:
                outcome = results.get(node.node_id)
                if not isinstance(outcome, dict):
                    raise RuntimeError(f"graph result missing {node.node_id}")
                node.submitted = True
                node.outcome = outcome
        values = [self._value(target) for target in targets]
        return values[0] if len(values) == 1 else values

    def _closure(self, targets: tuple[ToolNode, ...]) -> list[ToolNode]:
        included: set[str] = set()
        visiting: set[str] = set()

        def visit(node: ToolNode) -> None:
            if node.node_id in included or node.submitted:
                return
            if node.node_id in visiting:
                raise RuntimeError("CYCLIC_DEPENDENCY")
            visiting.add(node.node_id)
            for dependency_id in node.dependencies:
                dependency = self.nodes.get(dependency_id)
                if dependency is None:
                    raise RuntimeError("UNKNOWN_NODE_REFERENCE")
                visit(dependency)
            visiting.remove(node.node_id)
            included.add(node.node_id)

        for target in targets:
            visit(target)
        return sorted(
            (self.nodes[node_id] for node_id in included),
            key=lambda node: node.ordinal,
        )

    def _value(self, node: ToolNode) -> Any:
        outcome = node.outcome
        if not isinstance(outcome, dict):
            raise RuntimeError("node has no outcome")
        if outcome.get("ok") is True:
            return outcome.get("value")
        error = outcome.get("error")
        if not isinstance(error, dict):
            error = {"message": "tool call failed"}
        error_type = (
            GraphExecutionError
            if error.get("code") in {"UPSTREAM_FAILED", "GRAPH_REJECTED"}
            else ToolCallError
        )
        raise error_type(node.name, node.node_id, error)


def _send(value: dict[str, Any]) -> None:
    sys.stdout.buffer.write(encode_frame(value))
    sys.stdout.buffer.flush()


def _read() -> dict[str, Any]:
    raw = sys.stdin.buffer.readline(1024 * 1024 + 1)
    if not raw:
        raise EOFError("parent closed protocol")
    return decode_frame(raw)


async def _main(request: dict[str, Any]) -> dict[str, Any]:
    source = request.get("code")
    tools = request.get("tools")
    if not isinstance(source, str) or not isinstance(tools, list) or any(
        not isinstance(name, str) for name in tools
    ):
        raise CodeValidationError("invalid initial request")
    _apply_resource_limits(request)
    validated = validate_code(source, max_nodes=int(request["max_ast_nodes"]))
    client = _GraphClient(tuple(tools))
    output = _ReturnStream()

    async def gather(*nodes: ToolNode) -> list[Any]:
        value = await client.execute(tuple(nodes))
        return value if isinstance(value, list) else [value]

    async def execute(*nodes: ToolNode) -> list[Any]:
        return await gather(*nodes)

    environment: dict[str, Any] = {
        "tools": _Tools(client, tuple(tools)),
        "gather": gather,
        "execute": execute,
        "print": output.print,
        "ToolCallError": ToolCallError,
        "GraphExecutionError": GraphExecutionError,
        "len": len, "range": range, "enumerate": enumerate, "zip": zip,
        "min": min, "max": max, "sum": sum, "sorted": sorted,
        "any": any, "all": all, "abs": abs, "round": round,
        "str": str, "int": int, "float": float, "bool": bool,
        "list": list, "dict": dict, "True": True, "False": False, "None": None,
    }
    def bounded_range(*args: int) -> range:
        value = range(*args)
        if len(value) > int(request["max_collection_items"]):
            raise CollectionLimitError("collection item limit exceeded")
        return value

    environment["range"] = bounded_range
    value = await Evaluator(
        environment,
        max_steps=int(request["max_eval_steps"]),
        max_collection_items=int(request["max_collection_items"]),
    ).run(validated)
    return {"type": "done", "value": ensure_json_value(value), "logs": output.logs}


def _apply_resource_limits(
    request: dict[str, Any], *, resource_module: Any = None
) -> None:
    """Tighten worker CPU/address-space soft limits when POSIX supports it."""

    limits = _resource if resource_module is None else resource_module
    if limits is None:
        return
    cpu_seconds = request.get("max_cpu_seconds")
    memory_mb = request.get("max_memory_mb")
    if type(cpu_seconds) is not int or cpu_seconds < 1:
        raise CodeValidationError("max_cpu_seconds must be a positive integer")
    if type(memory_mb) is not int or memory_mb < 1:
        raise CodeValidationError("max_memory_mb must be a positive integer")
    _tighten_limit(limits, limits.RLIMIT_CPU, cpu_seconds)
    if hasattr(limits, "RLIMIT_AS"):
        try:
            _tighten_limit(limits, limits.RLIMIT_AS, memory_mb * 1024 * 1024)
        except (OSError, ValueError):
            # Darwin exposes RLIMIT_AS but rejects changing it. The short-lived
            # worker still has empty env/cwd plus wall, AST, step, collection,
            # protocol, and output limits; Linux keeps the address-space cap.
            pass


def _tighten_limit(limits: Any, kind: int, requested: int) -> None:
    current_soft, current_hard = limits.getrlimit(kind)
    infinity = limits.RLIM_INFINITY
    soft = (
        requested
        if current_soft == infinity
        else min(current_soft, requested)
    )
    limits.setrlimit(kind, (soft, current_hard))


def main() -> int:
    try:
        request = _read()
        result = asyncio.run(_main(request))
    except EvaluationLimitError as error:
        result = {"type": "done", "error": {"code": "EVAL_STEP_LIMIT", "message": str(error)}, "logs": []}
    except CollectionLimitError as error:
        result = {"type": "done", "error": {"code": "COLLECTION_LIMIT", "message": str(error)}, "logs": []}
    except CodeValidationError as error:
        result = {"type": "done", "error": {"code": "INVALID_CODE", "message": str(error)}, "logs": []}
    except BaseException as error:
        result = {"type": "done", "error": {"code": "CODE_EXCEPTION", "message": f"{type(error).__name__}: {error}"}, "logs": []}
    _send(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
