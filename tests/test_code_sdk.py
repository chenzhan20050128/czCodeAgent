"""Model-facing Python SDK generation for run_code."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mca.code_sdk import render_python_sdk
from mca.tools import create_tool_registry


class CodeSdkTests(unittest.TestCase):
    def test_sdk_exposes_six_tools_but_not_control_or_recursive_tools(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            text = render_python_sdk(create_tool_registry(temporary))

        for name in ("read_file", "list_dir", "grep", "write_file", "edit_file", "bash"):
            self.assertIn(f"def {name}(", text)
        self.assertNotIn("def exit_plan_mode(", text)
        self.assertNotIn("def run_code(", text)
        self.assertIn("after: list[ToolNode] | None", text)
        self.assertIn("await gather", text)
        self.assertIn("UPSTREAM_FAILED", text)
        self.assertIn('ToolResult = {"status": str, "output": str', text)
        self.assertIn('readme_text = results[0]["output"]', text)
        self.assertIn("return {", text)
        self.assertIn("Imports and json.dumps/json.dump are unavailable", text)
        self.assertIn("A final bare expression is discarded", text)

    def test_run_code_description_contains_the_generated_sdk(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            registry = create_tool_registry(temporary)
        description = registry.resolve("run_code").description
        self.assertIn("Constrained Python", description)
        self.assertIn("def write_file(", description)
        self.assertNotIn("def exit_plan_mode(", description)


if __name__ == "__main__":
    unittest.main()
