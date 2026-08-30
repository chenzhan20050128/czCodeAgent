"""Tests for explicit tool contracts and bounded results."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mca.tools.registry import (
    ExecutionMode,
    SideEffect,
    ToolRegistry,
    ToolResult,
    ToolSpec,
    ToolValidationError,
    UnknownToolError,
    truncate_output,
)
from mca.tools import create_tool_registry


class ToolRegistryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.schema = {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "offset": {"type": "integer", "minimum": 1},
                "ratio": {"type": "number", "maximum": 1.0},
                "enabled": {"type": "boolean"},
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "mode": {"type": "string", "enum": ["fast", "safe"]},
                "selector": {
                    "anyOf": [
                        {"type": "string"},
                        {"type": "integer"},
                    ]
                },
            },
            "required": ["path", "offset"],
            "additionalProperties": False,
        }
        self.spec = ToolSpec(
            name="read_file",
            description="Read a workspace text file.",
            schema=self.schema,
            handler=lambda arguments: arguments,
            side_effect=SideEffect.NONE,
        )
        self.registry = ToolRegistry([self.spec])

    def test_provider_schema_has_standard_function_shape(self) -> None:
        self.assertEqual(
            self.spec.provider_schema(),
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read a workspace text file.",
                    "parameters": self.schema,
                },
            },
        )

    def test_registry_advertises_all_provider_schemas(self) -> None:
        self.assertEqual(self.registry.provider_schemas(), [self.spec.provider_schema()])

    def test_parse_and_validate_returns_object_arguments(self) -> None:
        raw = json.dumps(
            {
                "path": "src/main.py",
                "offset": 1,
                "ratio": 0.5,
                "enabled": True,
                "tags": ["python"],
                "mode": "safe",
                "selector": "name",
            }
        )

        parsed = self.registry.parse_and_validate("read_file", raw)

        self.assertEqual(parsed["path"], "src/main.py")
        self.assertEqual(parsed["tags"], ["python"])

    def test_unknown_tool_raises_stable_error(self) -> None:
        with self.assertRaisesRegex(UnknownToolError, "unknown tool: missing"):
            self.registry.parse_and_validate("missing", "{}")

    def test_malformed_json_raises_stable_error(self) -> None:
        with self.assertRaisesRegex(ToolValidationError, "arguments must be valid JSON"):
            self.registry.parse_and_validate("read_file", "{")

    def test_non_object_arguments_are_rejected(self) -> None:
        with self.assertRaisesRegex(ToolValidationError, "arguments must be an object"):
            self.registry.parse_and_validate("read_file", "[]")

    def test_missing_required_property_is_rejected(self) -> None:
        with self.assertRaisesRegex(ToolValidationError, "missing required property: offset"):
            self.registry.parse_and_validate("read_file", '{"path": "a"}')

    def test_unknown_property_is_rejected(self) -> None:
        raw = '{"path": "a", "offset": 1, "surprise": true}'
        with self.assertRaisesRegex(ToolValidationError, "unknown property: surprise"):
            self.registry.parse_and_validate("read_file", raw)

    def test_primitive_types_are_checked(self) -> None:
        cases = [
            ('{"path": 3, "offset": 1}', "path must be a string"),
            ('{"path": "a", "offset": 1, "enabled": 1}', "enabled must be a boolean"),
            ('{"path": "a", "offset": 1, "tags": "x"}', "tags must be an array"),
        ]
        for raw, message in cases:
            with self.subTest(raw=raw):
                with self.assertRaisesRegex(ToolValidationError, message):
                    self.registry.parse_and_validate("read_file", raw)

    def test_bool_is_not_accepted_as_integer_or_number(self) -> None:
        for field in ("offset", "ratio"):
            raw = json.dumps({"path": "a", "offset": 1, field: True})
            with self.subTest(field=field):
                with self.assertRaisesRegex(ToolValidationError, f"{field} must be a"):
                    self.registry.parse_and_validate("read_file", raw)

    def test_numeric_bounds_are_checked(self) -> None:
        cases = [
            ('{"path": "a", "offset": 0}', "offset must be >= 1"),
            ('{"path": "a", "offset": 1, "ratio": 1.5}', "ratio must be <= 1.0"),
        ]
        for raw, message in cases:
            with self.subTest(raw=raw):
                with self.assertRaisesRegex(ToolValidationError, message):
                    self.registry.parse_and_validate("read_file", raw)

    def test_array_items_enum_and_any_of_are_checked(self) -> None:
        cases = [
            ({"path": "a", "offset": 1, "tags": [2]}, r"tags\[0\] must be a string"),
            ({"path": "a", "offset": 1, "mode": "turbo"}, "mode must be one of"),
            ({"path": "a", "offset": 1, "selector": False}, "selector does not match any allowed schema"),
        ]
        for value, message in cases:
            with self.subTest(value=value):
                with self.assertRaisesRegex(ToolValidationError, message):
                    self.registry.parse_and_validate("read_file", json.dumps(value))

    def test_duplicate_tool_names_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "duplicate tool: read_file"):
            ToolRegistry([self.spec, self.spec])

    def test_spec_requires_exactly_one_handler_kind(self) -> None:
        with self.assertRaisesRegex(ValueError, "exactly one"):
            ToolSpec("x", "desc", self.schema, side_effect=SideEffect.NONE)
        with self.assertRaisesRegex(ValueError, "exactly one"):
            ToolSpec(
                "x",
                "desc",
                self.schema,
                handler=lambda arguments: arguments,
                prepare_handler=lambda arguments: arguments,
                side_effect=SideEffect.NONE,
            )

    def test_execution_mode_requires_explicit_safe_side_effect_free_call(self) -> None:
        safe = ToolSpec(
            name="safe",
            description="Safe read.",
            schema={
                "type": "object",
                "properties": {"mode": {"type": "string"}},
                "required": ["mode"],
                "additionalProperties": False,
            },
            handler=lambda arguments: arguments,
            side_effect=SideEffect.NONE,
            is_concurrency_safe=lambda arguments: arguments["mode"] == "read",
        )
        write = ToolSpec(
            name="write",
            description="Unsafe write.",
            schema={
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False,
            },
            prepare_handler=lambda arguments: arguments,
            side_effect=SideEffect.WORKSPACE_WRITE,
            is_concurrency_safe=lambda arguments: True,
        )
        registry = ToolRegistry([safe, write])

        self.assertIs(
            registry.execution_mode("safe", '{"mode":"read"}'),
            ExecutionMode.PARALLEL,
        )
        for name, arguments in (
            ("safe", '{"mode":"write"}'),
            ("safe", "{}"),
            ("safe", "{"),
            ("write", "{}"),
            ("missing", "{}"),
        ):
            with self.subTest(name=name, arguments=arguments):
                self.assertIs(
                    registry.execution_mode(name, arguments),
                    ExecutionMode.EXCLUSIVE,
                )

    def test_concurrency_classifier_failure_is_exclusive_and_not_model_visible(self) -> None:
        def explode(arguments: dict[str, object]) -> bool:
            raise RuntimeError("classifier failed")

        spec = ToolSpec(
            name="safe",
            description="Safe read.",
            schema={
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False,
            },
            handler=lambda arguments: arguments,
            side_effect=SideEffect.NONE,
            is_concurrency_safe=explode,
        )
        registry = ToolRegistry([spec])

        self.assertIs(
            registry.execution_mode("safe", "{}"),
            ExecutionMode.EXCLUSIVE,
        )
        self.assertEqual(
            set(spec.provider_schema()["function"]),
            {"name", "description", "parameters"},
        )


class ToolResultTruncationTests(unittest.TestCase):
    def test_short_output_is_unchanged(self) -> None:
        output, truncated = truncate_output("one\ntwo", max_bytes=100, max_lines=3)
        self.assertEqual(output, "one\ntwo")
        self.assertFalse(truncated)

    def test_line_limit_keeps_head_and_tail_with_marker(self) -> None:
        output, truncated = truncate_output(
            "one\ntwo\nthree\nfour\nfive", max_bytes=100, max_lines=4
        )

        self.assertTrue(truncated)
        self.assertEqual(output.splitlines()[0], "one")
        self.assertEqual(output.splitlines()[-1], "five")
        self.assertIn("truncated", output)
        self.assertLessEqual(len(output.splitlines()), 4)

    def test_utf8_byte_limit_keeps_head_and_tail_without_splitting_codepoints(self) -> None:
        output, truncated = truncate_output(
            "甲乙丙丁戊己庚辛壬癸", max_bytes=29, max_lines=5
        )

        self.assertTrue(truncated)
        self.assertTrue(output.startswith("甲"))
        self.assertTrue(output.endswith("癸"))
        self.assertIn("truncated", output)
        self.assertLessEqual(len(output.encode("utf-8")), 29)

    def test_tool_result_bounded_records_truncation_in_metadata(self) -> None:
        result = ToolResult.bounded(
            title="search",
            output="a\nb\nc\nd",
            status="succeeded",
            metadata={"matches": 4},
            max_bytes=100,
            max_lines=3,
        )

        self.assertEqual(result.title, "search")
        self.assertEqual(result.status, "succeeded")
        self.assertEqual(result.metadata["matches"], 4)
        self.assertIs(result.metadata["truncated"], True)
        self.assertLessEqual(len(result.output.splitlines()), 3)


class BuiltinToolSpecTests(unittest.TestCase):
    def test_registry_contains_the_documented_tools_and_plan_exit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            registry = create_tool_registry(temporary)

        schemas = registry.provider_schemas()
        self.assertEqual(
            [schema["function"]["name"] for schema in schemas],
            [
                "read_file",
                "list_dir",
                "grep",
                "write_file",
                "edit_file",
                "bash",
                "exit_plan_mode",
                "run_code",
            ],
        )
        for schema in schemas:
            function = schema["function"]
            self.assertTrue(function["description"])
            self.assertEqual(function["parameters"]["type"], "object")
            self.assertIs(function["parameters"]["additionalProperties"], False)

    def test_file_and_shell_handlers_are_bound(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            registry = create_tool_registry(temporary)

        self.assertIsNotNone(registry.resolve("read_file").handler)
        self.assertIsNotNone(registry.resolve("write_file").prepare_handler)
        prepared = registry.resolve("bash").prepare_handler({"command": "pwd"})
        self.assertEqual(prepared.command, "pwd")

    def test_required_path_arguments_reject_empty_strings(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            registry = create_tool_registry(temporary)

        for tool, payload in (
            ("read_file", '{"path": ""}'),
            ("write_file", '{"path": "", "content": "x"}'),
            ("edit_file", '{"path": "", "old_text": "a", "new_text": "b"}'),
            ("grep", '{"pattern": ""}'),
        ):
            with self.subTest(tool=tool):
                with self.assertRaises(ToolValidationError):
                    registry.parse_and_validate(tool, payload)

    def test_write_file_description_discourages_manual_mkdir(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            registry = create_tool_registry(temporary)

        description = registry.resolve("write_file").description
        self.assertIn("parent directories are created", description)

    def test_only_builtin_read_and_list_calls_are_parallel_safe(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            registry = create_tool_registry(temporary)
            expected = {
                "read_file": ExecutionMode.PARALLEL,
                "list_dir": ExecutionMode.PARALLEL,
                "grep": ExecutionMode.EXCLUSIVE,
                "write_file": ExecutionMode.EXCLUSIVE,
                "edit_file": ExecutionMode.EXCLUSIVE,
                "bash": ExecutionMode.EXCLUSIVE,
                "exit_plan_mode": ExecutionMode.EXCLUSIVE,
                "run_code": ExecutionMode.EXCLUSIVE,
            }
            arguments = {
                "read_file": '{"path":"missing.txt"}',
                "list_dir": "{}",
                "grep": '{"pattern":"x"}',
                "write_file": '{"path":"a","content":"x"}',
                "edit_file": '{"path":"a","old_text":"x","new_text":"y"}',
                "bash": '{"command":"true"}',
                "exit_plan_mode": '{"plan":"# Plan"}',
                "run_code": '{"description":"work","code":"return 1"}',
            }

            self.assertEqual(
                {name: registry.execution_mode(name, arguments[name]) for name in expected},
                expected,
            )


if __name__ == "__main__":
    unittest.main()
