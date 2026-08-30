"""Validation and evaluation of MCA's constrained async Python subset."""

from __future__ import annotations

import ast
import json
import operator
from dataclasses import dataclass
from typing import Any, NoReturn


class CodeValidationError(ValueError):
    """The submitted source is outside the supported Python subset."""


class EvaluationLimitError(RuntimeError):
    """The interpreter exhausted its deterministic step budget."""


class CollectionLimitError(RuntimeError):
    """The program attempted to materialize too many collection items."""


class ToolCallError(RuntimeError):
    def __init__(self, tool_name: str, node_id: str, error: dict[str, Any]) -> None:
        super().__init__(str(error.get("message", "tool call failed")))
        self.tool_name = tool_name
        self.node_id = node_id
        self.status = str(error.get("status", "failed"))
        self.code = str(error.get("code", "TOOL_FAILED"))
        self.details = dict(error)


class GraphExecutionError(ToolCallError):
    pass


@dataclass(frozen=True)
class ValidatedCode:
    source: str
    body: tuple[ast.stmt, ...]
    node_count: int


_FORBIDDEN = (
    ast.Import, ast.ImportFrom, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef,
    ast.Lambda, ast.Global, ast.Nonlocal, ast.With, ast.AsyncWith, ast.Delete,
    ast.Yield, ast.YieldFrom, ast.NamedExpr, ast.Match,
)
_ALLOWED_NAMES = {
    "tools", "gather", "execute", "print", "ToolCallError",
    "GraphExecutionError", "len", "range", "enumerate", "zip",
    "min", "max", "sum", "sorted", "any", "all", "abs",
    "round", "str", "int", "float", "bool", "list", "dict",
    "True", "False", "None",
}
_SAFE_METHODS = {
    "get", "keys", "values", "items", "split", "splitlines",
    "startswith", "endswith", "lower", "upper", "strip",
    "replace", "join", "append", "extend", "after",
}


def validate_code(source: str, *, max_nodes: int = 10_000) -> ValidatedCode:
    if not isinstance(source, str) or not source.strip():
        raise CodeValidationError("code must be a non-empty string")
    if type(max_nodes) is not int or max_nodes < 1:
        raise ValueError("max_nodes must be positive")
    wrapped = "async def __mca_main__():\n" + "".join(
        f"    {line}\n" for line in source.splitlines()
    )
    try:
        module = ast.parse(wrapped, mode="exec")
    except SyntaxError as error:
        line = max(1, (error.lineno or 2) - 1)
        raise CodeValidationError(
            f"invalid code at line {line}: {error.msg}"
        ) from None
    function = module.body[0]
    assert isinstance(function, ast.AsyncFunctionDef)
    nodes = [node for statement in function.body for node in ast.walk(statement)]
    if len(nodes) > max_nodes:
        raise CodeValidationError("code exceeds AST node limit")
    for node in nodes:
        if isinstance(node, _FORBIDDEN):
            raise CodeValidationError(
                f"unsupported syntax: {type(node).__name__}"
            )
        if isinstance(node, ast.Name) and node.id.startswith("_"):
            raise CodeValidationError("private names are not allowed")
        if isinstance(node, ast.Attribute):
            if node.attr.startswith("_"):
                raise CodeValidationError("private attributes are not allowed")
            if not (
                isinstance(node.value, ast.Name) and node.value.id == "tools"
            ) and node.attr not in _SAFE_METHODS and node.attr not in {
                "tool_name", "node_id", "status", "code", "message", "details"
            }:
                raise CodeValidationError(
                    f"unsupported attribute: {node.attr}"
                )
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                if node.func.id not in _ALLOWED_NAMES:
                    raise CodeValidationError(
                        f"unsupported call: {node.func.id}"
                    )
            elif isinstance(node.func, ast.Attribute):
                if not (
                    isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "tools"
                ) and node.func.attr not in _SAFE_METHODS:
                    raise CodeValidationError(
                        f"unsupported method call: {node.func.attr}"
                    )
            else:
                raise CodeValidationError("dynamic calls are not allowed")
    return ValidatedCode(source, tuple(function.body), len(nodes))


class _ReturnSignal(BaseException):
    def __init__(self, value: Any) -> None:
        self.value = value


class _BreakSignal(BaseException):
    pass


class _ContinueSignal(BaseException):
    pass


class Evaluator:
    """Small async AST interpreter with deterministic operation bounds."""

    def __init__(
        self, environment: dict[str, Any], *, max_steps: int, max_collection_items: int = 10_000
    ) -> None:
        self.environment = environment
        self.max_steps = max_steps
        self.max_collection_items = max_collection_items
        self.steps = 0

    def tick(self) -> None:
        self.steps += 1
        if self.steps > self.max_steps:
            raise EvaluationLimitError("evaluation step limit exceeded")

    async def run(self, code: ValidatedCode) -> Any:
        try:
            await self._statements(code.body)
        except _ReturnSignal as returned:
            return returned.value
        return None

    async def _statements(self, statements: tuple[ast.stmt, ...] | list[ast.stmt]) -> None:
        for statement in statements:
            self.tick()
            await self._statement(statement)

    async def _statement(self, node: ast.stmt) -> None:
        if isinstance(node, ast.Assign):
            value = await self._expr(node.value)
            for target in node.targets:
                self._assign(target, value)
            return
        if isinstance(node, ast.AnnAssign):
            if node.value is not None:
                self._assign(node.target, await self._expr(node.value))
            return
        if isinstance(node, ast.Expr):
            await self._expr(node.value)
            return
        if isinstance(node, ast.Return):
            raise _ReturnSignal(
                None if node.value is None else await self._expr(node.value)
            )
        if isinstance(node, ast.If):
            branch = node.body if await self._expr(node.test) else node.orelse
            await self._statements(branch)
            return
        if isinstance(node, (ast.For, ast.AsyncFor)):
            iterable = await self._expr(node.iter)
            for index, item in enumerate(iterable):
                if index >= self.max_collection_items:
                    raise CollectionLimitError("collection item limit exceeded")
                self.tick()
                self._assign(node.target, item)
                try:
                    await self._statements(node.body)
                except _ContinueSignal:
                    continue
                except _BreakSignal:
                    break
            else:
                await self._statements(node.orelse)
            return
        if isinstance(node, ast.While):
            while await self._expr(node.test):
                self.tick()
                try:
                    await self._statements(node.body)
                except _ContinueSignal:
                    continue
                except _BreakSignal:
                    break
            else:
                await self._statements(node.orelse)
            return
        if isinstance(node, ast.Try):
            try:
                await self._statements(node.body)
            except (ToolCallError, GraphExecutionError) as error:
                for handler in node.handlers:
                    if handler.type is None:
                        matched = True
                    elif isinstance(handler.type, ast.Name):
                        exception_type = self.environment.get(handler.type.id)
                        matched = (
                            exception_type in {ToolCallError, GraphExecutionError}
                            and isinstance(error, exception_type)
                        )
                    else:
                        matched = False
                    if matched:
                        if handler.name:
                            self.environment[handler.name] = error
                        await self._statements(handler.body)
                        break
                else:
                    raise
            else:
                await self._statements(node.orelse)
            finally:
                await self._statements(node.finalbody)
            return
        if isinstance(node, ast.Pass):
            return
        if isinstance(node, ast.Break):
            raise _BreakSignal
        if isinstance(node, ast.Continue):
            raise _ContinueSignal
        raise CodeValidationError(f"unsupported statement: {type(node).__name__}")

    def _assign(self, target: ast.expr, value: Any) -> None:
        if isinstance(target, ast.Name):
            self.environment[target.id] = value
            return
        if isinstance(target, (ast.Tuple, ast.List)):
            values = list(value)
            if len(values) != len(target.elts):
                raise ValueError("unpack length mismatch")
            for child, item in zip(target.elts, values, strict=True):
                self._assign(child, item)
            return
        if isinstance(target, ast.Subscript):
            container = self._sync_expr(target.value)
            key = self._sync_expr(target.slice)
            container[key] = value
            self._bounded_collection(container)
            return
        raise CodeValidationError("unsupported assignment target")

    async def _expr(self, node: ast.expr) -> Any:
        self.tick()
        if isinstance(node, ast.Await):
            value = await self._expr(node.value)
            return await value
        if isinstance(node, ast.Call):
            function = await self._expr(node.func)
            args = [await self._expr(arg) for arg in node.args]
            kwargs = {item.arg: await self._expr(item.value) for item in node.keywords if item.arg}
            return self._invoke_bounded(function, args, kwargs)
        if isinstance(node, ast.ListComp):
            return await self._list_comp(node)
        if isinstance(node, ast.DictComp):
            values = await self._comprehension(node.generators, node.key, node.value)
            return {key: value for key, value in values}
        if isinstance(node, ast.BoolOp):
            values = []
            for item in node.values:
                value = await self._expr(item)
                values.append(value)
                if isinstance(node.op, ast.And) and not value:
                    return value
                if isinstance(node.op, ast.Or) and value:
                    return value
            return values[-1]
        if isinstance(node, ast.IfExp):
            return await self._expr(node.body if await self._expr(node.test) else node.orelse)
        return self._sync_expr(node)

    def _sync_expr(self, node: ast.expr) -> Any:
        self.tick()
        if isinstance(node, ast.Call):
            function = self._sync_expr(node.func)
            args = [self._sync_expr(argument) for argument in node.args]
            kwargs = {
                item.arg: self._sync_expr(item.value)
                for item in node.keywords
                if item.arg is not None
            }
            return self._invoke_bounded(function, args, kwargs)
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name):
            if node.id not in self.environment:
                raise NameError(node.id)
            return self.environment[node.id]
        if isinstance(node, ast.List):
            if len(node.elts) > self.max_collection_items:
                raise CollectionLimitError("collection item limit exceeded")
            return [self._sync_expr(item) for item in node.elts]
        if isinstance(node, ast.Tuple):
            if len(node.elts) > self.max_collection_items:
                raise CollectionLimitError("collection item limit exceeded")
            return tuple(self._sync_expr(item) for item in node.elts)
        if isinstance(node, ast.Dict):
            if len(node.keys) > self.max_collection_items:
                raise CollectionLimitError("collection item limit exceeded")
            return {self._sync_expr(key): self._sync_expr(value) for key, value in zip(node.keys, node.values, strict=True)}
        if isinstance(node, ast.Attribute):
            value = self._sync_expr(node.value)
            if node.attr.startswith("_"):
                raise CodeValidationError("private attributes are not allowed")
            if isinstance(value, (ToolCallError, GraphExecutionError)) and node.attr in {"tool_name", "node_id", "status", "code", "message", "details"}:
                return str(value) if node.attr == "message" else getattr(value, node.attr)
            if node.attr in _SAFE_METHODS and type(value) in {str, list, dict}:
                return getattr(value, node.attr)
            return getattr(value, node.attr)
        if isinstance(node, ast.Subscript):
            return self._sync_expr(node.value)[self._sync_expr(node.slice)]
        if isinstance(node, ast.Slice):
            return slice(
                self._sync_expr(node.lower) if node.lower else None,
                self._sync_expr(node.upper) if node.upper else None,
                self._sync_expr(node.step) if node.step else None,
            )
        if isinstance(node, ast.BinOp):
            operations = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod}
            operation = operations.get(type(node.op))
            if operation is None:
                raise CodeValidationError("unsupported binary operator")
            left = self._sync_expr(node.left)
            right = self._sync_expr(node.right)
            if isinstance(node.op, ast.Mult):
                self._reject_oversized_repetition(left, right)
            return self._bounded_collection(
                operation(left, right)
            )
        if isinstance(node, ast.UnaryOp):
            value = self._sync_expr(node.operand)
            if isinstance(node.op, ast.Not): return not value
            if isinstance(node.op, ast.USub): return -value
            if isinstance(node.op, ast.UAdd): return +value
            raise CodeValidationError("unsupported unary operator")
        if isinstance(node, ast.Compare):
            left = self._sync_expr(node.left)
            operations = {ast.Eq: operator.eq, ast.NotEq: operator.ne, ast.Lt: operator.lt, ast.LtE: operator.le, ast.Gt: operator.gt, ast.GtE: operator.ge, ast.In: lambda a, b: a in b, ast.NotIn: lambda a, b: a not in b, ast.Is: operator.is_, ast.IsNot: operator.is_not}
            for op_node, comparator in zip(node.ops, node.comparators, strict=True):
                right = self._sync_expr(comparator)
                operation = operations.get(type(op_node))
                if operation is None or not operation(left, right):
                    return False
                left = right
            return True
        if isinstance(node, ast.JoinedStr):
            parts = []
            for item in node.values:
                if isinstance(item, ast.Constant):
                    parts.append(str(item.value))
                else:
                    assert isinstance(item, ast.FormattedValue)
                    parts.append(str(self._sync_expr(item.value)))
            return "".join(parts)
        raise CodeValidationError(f"unsupported expression: {type(node).__name__}")

    def _bounded_collection(self, value: Any) -> Any:
        if (
            type(value) in {str, list, tuple, dict, range}
            and len(value) > self.max_collection_items
        ):
            raise CollectionLimitError("collection item limit exceeded")
        return value

    def _invoke_bounded(
        self, function: Any, args: list[Any], kwargs: dict[str, Any]
    ) -> Any:
        result = function(*args, **kwargs)
        receiver = getattr(function, "__self__", None)
        if type(receiver) in {list, dict}:
            self._bounded_collection(receiver)
        return self._bounded_collection(result)

    def _reject_oversized_repetition(self, left: Any, right: Any) -> None:
        sequence: Any = None
        multiplier: Any = None
        if type(left) in {str, list, tuple} and type(right) is int:
            sequence, multiplier = left, right
        elif type(right) in {str, list, tuple} and type(left) is int:
            sequence, multiplier = right, left
        if sequence is not None and len(sequence) * max(0, multiplier) > self.max_collection_items:
            raise CollectionLimitError("collection item limit exceeded")

    async def _list_comp(self, node: ast.ListComp) -> list[Any]:
        return await self._comprehension(node.generators, node.elt)

    async def _comprehension(self, generators: list[ast.comprehension], *expressions: ast.expr) -> list[Any]:
        output: list[Any] = []

        async def visit(index: int) -> None:
            if index == len(generators):
                values = tuple(await self._expr(expression) for expression in expressions)
                output.append(values[0] if len(values) == 1 else values)
                return
            generator = generators[index]
            if generator.is_async:
                raise CodeValidationError("async comprehensions are not supported")
            iterable = await self._expr(generator.iter)
            for item in iterable:
                if len(output) >= self.max_collection_items:
                    raise CollectionLimitError("collection item limit exceeded")
                self.tick()
                self._assign(generator.target, item)
                if all(await self._expr(condition) for condition in generator.ifs):
                    await visit(index + 1)

        await visit(0)
        return output


def ensure_json_value(value: Any) -> Any:
    try:
        return json.loads(json.dumps(value, allow_nan=False))
    except (TypeError, ValueError):
        raise ValueError("program completion must be lossless JSON") from None


__all__ = ["CodeValidationError", "CollectionLimitError", "EvaluationLimitError", "Evaluator", "GraphExecutionError", "ToolCallError", "ValidatedCode", "ensure_json_value", "validate_code"]
