"""Constrained Python Code Mode runtime tests."""

from __future__ import annotations

import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mca.code_ast import CodeValidationError, validate_code
from mca.code_protocol import ProtocolFrameError, decode_frame, encode_frame
from mca.code_runtime import CodeRuntime, CodeRuntimeConfig
import mca.code_runtime as code_runtime
import mca.code_worker as code_worker


class CodeAstTests(unittest.TestCase):
    def test_accepts_async_dag_language(self) -> None:
        validate_code(
            """
a = tools.read_file({"path": "a.py"})
b = tools.write_file({"path": "b.py", "content": "x"})
test = tools.bash({"command": "python3 -m unittest"}, after=[a, b])
try:
    result = await test
except ToolCallError as error:
    result = {"code": error.code}
return {"result": result, "count": len([a, b])}
"""
        )

    def test_rejects_ambient_authority_and_reflection(self) -> None:
        cases = (
            "import os",
            "from pathlib import Path",
            "open('x')",
            "eval('1')",
            "getattr(tools, 'read_file')",
            "x.__class__",
            "def f():\n    pass",
            "class X:\n    pass",
            "f = lambda: 1",
        )
        for source in cases:
            with self.subTest(source=source):
                with self.assertRaises(CodeValidationError):
                    validate_code(source)

    def test_rejects_oversized_ast(self) -> None:
        source = "\n".join(f"x{i} = {i}" for i in range(10))
        with self.assertRaisesRegex(CodeValidationError, "AST node limit"):
            validate_code(source, max_nodes=5)


class CodeProtocolTests(unittest.TestCase):
    def test_frame_round_trip_is_canonical_jsonl(self) -> None:
        encoded = encode_frame({"type": "done", "value": {"你好": 1}})
        self.assertTrue(encoded.endswith(b"\n"))
        self.assertEqual(decode_frame(encoded), {"type": "done", "value": {"你好": 1}})

    def test_frame_rejects_non_object_non_json_and_oversize(self) -> None:
        for raw in (b"[]\n", b"not-json\n", b'{"type":NaN}\n'):
            with self.subTest(raw=raw):
                with self.assertRaises(ProtocolFrameError):
                    decode_frame(raw)
        with self.assertRaisesRegex(ProtocolFrameError, "too large"):
            decode_frame(b'{"type":"x","value":"123456789"}\n', max_bytes=10)


class CodeRuntimeTests(unittest.TestCase):
    def test_runtime_config_validates_cpu_and_memory_limits(self) -> None:
        config = CodeRuntimeConfig(max_cpu_seconds=7, max_memory_mb=384)
        self.assertEqual(config.max_cpu_seconds, 7)
        self.assertEqual(config.max_memory_mb, 384)
        for kwargs in ({"max_cpu_seconds": 0}, {"max_memory_mb": 0}):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    CodeRuntimeConfig(**kwargs)

    def test_start_frame_carries_worker_resource_budgets(self) -> None:
        captured: list[dict[str, object]] = []
        real_encode = encode_frame

        def recording_encode(value, **kwargs):
            captured.append(value)
            return real_encode(value, **kwargs)

        with patch("mca.code_runtime.encode_frame", side_effect=recording_encode):
            result = CodeRuntime(
                CodeRuntimeConfig(
                    max_wall_seconds=2, max_cpu_seconds=7, max_memory_mb=384
                )
            ).run("return 1", execute_graph=lambda request: {})

        self.assertIsNone(result.error)
        self.assertEqual(captured[0]["max_cpu_seconds"], 7)
        self.assertEqual(captured[0]["max_memory_mb"], 384)

    def test_worker_tightens_posix_cpu_and_address_space_limits(self) -> None:
        fake_resource = unittest.mock.Mock()
        fake_resource.RLIMIT_CPU = 1
        fake_resource.RLIMIT_AS = 2
        fake_resource.RLIM_INFINITY = -1
        fake_resource.getrlimit.side_effect = [(-1, -1), (-1, -1)]

        with patch.object(code_worker, "_resource", fake_resource, create=True):
            code_worker._apply_resource_limits(
                {"max_cpu_seconds": 7, "max_memory_mb": 384}
            )

        self.assertEqual(
            fake_resource.setrlimit.call_args_list,
            [
                unittest.mock.call(1, (7, -1)),
                unittest.mock.call(2, (384 * 1024 * 1024, -1)),
            ],
        )

    def test_worker_degrades_if_address_space_limit_is_unsupported(self) -> None:
        fake_resource = unittest.mock.Mock()
        fake_resource.RLIMIT_CPU = 1
        fake_resource.RLIMIT_AS = 2
        fake_resource.RLIM_INFINITY = -1
        fake_resource.getrlimit.side_effect = [(-1, -1), (-1, -1)]
        fake_resource.setrlimit.side_effect = [
            None,
            ValueError("current limit exceeds maximum limit"),
        ]

        code_worker._apply_resource_limits(
            {"max_cpu_seconds": 7, "max_memory_mb": 384},
            resource_module=fake_resource,
        )

        self.assertEqual(fake_resource.setrlimit.call_count, 2)

    def test_program_logs_are_bounded_by_utf8_output_bytes(self) -> None:
        result = CodeRuntime(
            CodeRuntimeConfig(max_wall_seconds=2, max_output_bytes=8)
        ).run(
            'print("你好你好")\nreturn 1',
            execute_graph=lambda request: {},
        )

        self.assertEqual(result.error.code, "OUTPUT_LIMIT")
        self.assertEqual(result.logs, ())

    def test_program_return_value_is_bounded_by_output_bytes(self) -> None:
        result = CodeRuntime(
            CodeRuntimeConfig(max_wall_seconds=2, max_output_bytes=32)
        ).run(
            'return {"payload": "x" * 100}',
            execute_graph=lambda request: {},
        )

        self.assertEqual(result.error.code, "OUTPUT_LIMIT")
        self.assertIsNone(result.value)

    def test_runs_program_and_exchanges_dynamic_graph(self) -> None:
        seen: list[dict[str, object]] = []

        def execute_graph(request: dict[str, object]) -> dict[str, object]:
            seen.append(request)
            nodes = request["nodes"]
            assert isinstance(nodes, list)
            results = {
                node["node_id"]: {
                    "ok": True,
                    "value": {"tool": node["name"], "arguments": node["arguments"]},
                }
                for node in nodes
            }
            return {"results": results}

        runtime = CodeRuntime(CodeRuntimeConfig(max_wall_seconds=2))
        result = runtime.run(
            """
a = tools.read_file({"path": "a.py"})
b = tools.write_file({"path": "b.py", "content": "x"})
test = tools.bash({"command": "test"}, after=[a, b])
values = await gather(a, b)
final = await test
print("completed", len(values))
return {"final": final}
""",
            execute_graph=execute_graph,
        )

        self.assertIsNone(result.error)
        self.assertEqual(result.logs, ("completed 2",))
        self.assertEqual(result.value["final"]["tool"], "bash")
        self.assertEqual(len(seen), 2)
        first_nodes = seen[0]["nodes"]
        self.assertEqual([node["name"] for node in first_nodes], ["read_file", "write_file"])
        second_nodes = seen[1]["nodes"]
        bash = next(node for node in second_nodes if node["name"] == "bash")
        self.assertEqual(len(bash["dependencies"]), 2)

    def test_safe_calls_work_inside_returned_json_composites(self) -> None:
        def execute_graph(request: dict[str, object]) -> dict[str, object]:
            targets = request["targets"]
            return {
                "results": {
                    target: {
                        "ok": True,
                        "value": {
                            "status": "succeeded",
                            "output": "Hello Agent",
                            "exit_code": None,
                            "truncated": False,
                            "metadata": {},
                        },
                    }
                    for target in targets
                }
            }

        result = CodeRuntime(CodeRuntimeConfig(max_wall_seconds=2)).run(
            """
node = tools.read_file({"path": "README.txt"})
tool_result = await node
text = tool_result["output"]
return {"length": len(text), "normalized": text.lower()}
""",
            execute_graph=execute_graph,
        )

        self.assertIsNone(result.error)
        self.assertEqual(
            result.value, {"length": 11, "normalized": "hello agent"}
        )

    def test_failed_target_raises_structured_graph_error_inside_program(self) -> None:
        def execute_graph(request: dict[str, object]) -> dict[str, object]:
            target = request["targets"][0]
            return {
                "results": {
                    target: {
                        "ok": False,
                        "error": {
                            "code": "UPSTREAM_FAILED",
                            "message": "dependency failed",
                            "status": "upstream_failed",
                            "blocked_by": ["node-1"],
                        },
                    }
                }
            }

        result = CodeRuntime(CodeRuntimeConfig(max_wall_seconds=2)).run(
            """
node = tools.bash({"command": "test"})
try:
    await node
except GraphExecutionError as error:
    return {"code": error.code, "blocked_by": error.details["blocked_by"]}
""",
            execute_graph=execute_graph,
        )

        self.assertEqual(result.value, {"code": "UPSTREAM_FAILED", "blocked_by": ["node-1"]})

    def test_cycle_is_rejected_before_graph_dispatch(self) -> None:
        calls = 0

        def execute_graph(request: dict[str, object]) -> dict[str, object]:
            nonlocal calls
            calls += 1
            return {}

        result = CodeRuntime(CodeRuntimeConfig(max_wall_seconds=2)).run(
            """
a = tools.read_file({"path": "a"})
b = tools.read_file({"path": "b"}, after=[a])
a.after(b)
return await a
""",
            execute_graph=execute_graph,
        )

        self.assertEqual(result.error.code, "CODE_EXCEPTION")
        self.assertIn("CYCLIC_DEPENDENCY", result.error.message)
        self.assertEqual(calls, 0)

    def test_parent_graph_validation_failure_is_structured_in_program(self) -> None:
        def execute_graph(request: dict[str, object]) -> dict[str, object]:
            raise ValueError("code graph exceeds node limit")

        result = CodeRuntime(CodeRuntimeConfig(max_wall_seconds=2)).run(
            """
node = tools.read_file({"path": "a"})
try:
    await node
except GraphExecutionError as error:
    return {"code": error.code, "message": error.message}
""",
            execute_graph=execute_graph,
        )

        self.assertEqual(result.value, {
            "code": "GRAPH_REJECTED",
            "message": "code graph exceeds node limit",
        })

    def test_runtime_has_empty_environment(self) -> None:
        secret_name = "MCA_CODE_RUNTIME_TEST_SECRET"
        with patch.dict(os.environ, {secret_name: "must-not-leak"}):
            result = CodeRuntime(CodeRuntimeConfig(max_wall_seconds=2)).run(
                "return await tools.read_file({\"path\": \"x\"})",
                execute_graph=lambda request: {"results": {request["targets"][0]: {"ok": True, "value": "ok"}}},
            )
        self.assertEqual(result.value, "ok")
        self.assertNotIn("must-not-leak", result.stderr)

    def test_hot_loop_hits_evaluation_budget(self) -> None:
        result = CodeRuntime(
            CodeRuntimeConfig(max_wall_seconds=2, max_eval_steps=20)
        ).run(
            "while True:\n    x = 1",
            execute_graph=lambda request: {},
        )
        self.assertEqual(result.error.code, "EVAL_STEP_LIMIT")

    def test_wall_timeout_kills_worker(self) -> None:
        result = CodeRuntime(
            CodeRuntimeConfig(max_wall_seconds=0.05, max_eval_steps=1_000_000)
        ).run(
            "while True:\n    pass",
            execute_graph=lambda request: {},
        )
        self.assertEqual(result.error.code, "WALL_TIMEOUT")

    def test_wall_timeout_is_global_across_protocol_frames(self) -> None:
        calls = 0

        def slow_graph(request: dict[str, object]) -> dict[str, object]:
            nonlocal calls
            calls += 1
            time.sleep(0.04)
            target = request["targets"][0]
            return {"results": {target: {"ok": True, "value": calls}}}

        result = CodeRuntime(
            CodeRuntimeConfig(max_wall_seconds=0.07)
        ).run(
            """
a = await tools.read_file({"path": "a"})
b = await tools.read_file({"path": "b"})
return [a, b]
""",
            execute_graph=slow_graph,
        )

        self.assertEqual(result.error.code, "WALL_TIMEOUT")

    def test_collection_limit_bounds_range_and_comprehension(self) -> None:
        result = CodeRuntime(
            CodeRuntimeConfig(max_wall_seconds=2, max_collection_items=3)
        ).run(
            "return [item for item in range(10)]",
            execute_graph=lambda request: {},
        )

        self.assertEqual(result.error.code, "COLLECTION_LIMIT")


if __name__ == "__main__":
    unittest.main()
