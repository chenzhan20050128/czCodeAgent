"""Tests for explicit tool contracts and bounded results."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mca.tools.registry import (
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
    def test_registry_contains_exactly_the_six_documented_tools(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            registry = create_tool_registry(temporary)

        schemas = registry.provider_schemas()
        self.assertEqual(
            [schema["function"]["name"] for schema in schemas],
            ["read_file", "list_dir", "grep", "write_file", "edit_file", "bash"],
        )
        for schema in schemas:
            function = schema["function"]
            self.assertTrue(function["description"])
            self.assertEqual(function["parameters"]["type"], "object")
            self.assertIs(function["parameters"]["additionalProperties"], False)

    def test_file_handlers_are_bound_and_shell_is_only_a_task6_placeholder(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            registry = create_tool_registry(temporary)

        self.assertIsNotNone(registry.resolve("read_file").handler)
        self.assertIsNotNone(registry.resolve("write_file").prepare_handler)
        with self.assertRaisesRegex(NotImplementedError, "Task 6"):
            registry.resolve("bash").prepare_handler({"command": "pwd"})


if __name__ == "__main__":
    unittest.main()
