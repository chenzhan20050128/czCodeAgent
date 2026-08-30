"""Tests for the mca command-line entry points and REPL wiring.

These are deterministic: the model client is replaced with a scripted fake so
the CLI is exercised without any network access.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import subprocess
import sys
import unittest
import uuid
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from mca import cli
from mca.domain import Event, SamplingOutcome, SessionReducer, SessionState
from mca.model import SampledToolCall, SamplingResult
from mca.store import RolloutStore


class ScriptedModel:
    """A stand-in ModelClient that yields queued sampling results."""

    def __init__(self, *results: SamplingResult) -> None:
        self._results = list(results)
        self.closed = False

    def sample(
        self,
        messages,
        tools,
        allow_tools,
        *,
        on_content=None,
        on_invalidate=None,
        on_reasoning=None,
    ):
        if not self._results:
            raise AssertionError("unexpected extra model sample")
        return self._results.pop(0)

    def close(self) -> None:
        self.closed = True


class StreamingModel(ScriptedModel):
    """A scripted model that emits its final text through the live callback."""

    def sample(
        self,
        messages,
        tools,
        allow_tools,
        *,
        on_content=None,
        on_invalidate=None,
        on_reasoning=None,
    ):
        result = super().sample(
            messages,
            tools,
            allow_tools,
            on_content=on_content,
            on_invalidate=on_invalidate,
            on_reasoning=on_reasoning,
        )
        if on_content is not None and result.content:
            on_content(result.content)
        return result


def _text(content: str) -> SamplingResult:
    return SamplingResult(
        SamplingOutcome.COMPLETE_TEXT, content=content, finish_reason="stop"
    )


class CliHelpTests(unittest.TestCase):
    def test_main_help_exits_zero_without_api_key(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            with contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(SystemExit) as raised:
                    cli.main(["--help"])
        self.assertEqual(raised.exception.code, 0)

    def test_python_module_help_exits_zero_without_api_key(self) -> None:
        environment = os.environ.copy()
        environment.pop("MCA_API_KEY", None)
        environment.pop("DEEPSEEK_API_KEY", None)
        environment["PYTHONPATH"] = str(SRC_ROOT)
        result = subprocess.run(
            [sys.executable, "-m", "mca", "--help"],
            cwd=PROJECT_ROOT,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("usage:", result.stdout)


class ConsoleFormattingTests(unittest.TestCase):
    def _code_events(self, *, description="update service", path="a.txt"):
        session_id = str(uuid.uuid4())
        turn_id = str(uuid.uuid4())
        run_id = str(uuid.uuid4())
        state = SessionState()
        events: list[Event] = []

        def apply(event_type, payload):
            event = Event.create(
                seq=len(events) + 1, session_id=session_id,
                event_type=event_type, payload=payload,
            )
            SessionReducer.apply(state, event)
            events.append(event)
            return event

        apply("session_created", {"cwd": "/work", "model": "m", "context_window": 4096})
        apply("turn_started", {"turn_id": turn_id, "user_input": "work"})
        apply("assistant_accepted", {"content": None, "tool_calls": [{"id": "outer", "type": "function", "function": {"name": "run_code", "arguments": "{}"}}]})
        apply("tool_started", {"call_key": "3:outer", "call_id": "outer"})
        started = apply("code_run_started", {"run_id": run_id, "turn_id": turn_id, "parent_call_key": "3:outer", "description": description, "source_hash": "sha256:x"})
        node_id = f"{run_id}:node:1"
        planned = apply("code_node_planned", {"run_id": run_id, "node_id": node_id, "ordinal": 1, "name": "write_file", "arguments": json.dumps({"path": path, "content": "x"}), "dependencies": []})
        return state, apply, started, planned, node_id

    def test_reasoning_is_shown_without_verbose_mode(self) -> None:
        captured = io.StringIO()
        console = cli._Console(verbose=False, color=False)

        with contextlib.redirect_stdout(captured):
            console.reasoning("inspect\nrepository")

        self.assertEqual(captured.getvalue(), "[thinking] inspect\nrepository")

    def test_verbose_reasoning_preserves_lines_and_separates_tool_calls(self) -> None:
        captured = io.StringIO()
        console = cli._Console(verbose=True, color=False)
        call = SampledToolCall(0, "c1", "function", "grep", '{"pattern":"FIXME"}')

        with contextlib.redirect_stdout(captured):
            console.reasoning("inspect\nrepository")
            console.tool_calls((call,))

        self.assertEqual(
            captured.getvalue(),
            '[thinking] inspect\nrepository\n\n[tool call] grep {"pattern":"FIXME"}\n',
        )

    def test_tool_approval_and_assistant_are_separate_blocks(self) -> None:
        captured = io.StringIO()
        console = cli._Console(verbose=True, color=False)
        call = SampledToolCall(0, "c1", "function", "bash", '{"command":"true"}')

        with contextlib.redirect_stdout(captured):
            console.tool_calls((call,))
            console.approval("Tool: bash\nAllow?")
            console.stream("Tests pass.")
            self.assertTrue(console.final_text_was_streamed("Tests pass."))

        self.assertEqual(
            captured.getvalue(),
            '[tool call] bash {"command":"true"}\n'
            '\n[approval] Tool: bash\nAllow?\n'
            '\nTests pass.\n',
        )

    def test_verbose_reasoning_and_assistant_are_separate_blocks(self) -> None:
        captured = io.StringIO()
        console = cli._Console(verbose=True, color=False)

        with contextlib.redirect_stdout(captured):
            console.reasoning("checking")
            console.stream("Ready.")
            self.assertTrue(console.final_text_was_streamed("Ready."))

        self.assertEqual(captured.getvalue(), "[thinking] checking\n\nReady.\n")

    def test_approval_and_following_reasoning_are_separate_blocks(self) -> None:
        captured = io.StringIO()
        console = cli._Console(verbose=True, color=False)

        with contextlib.redirect_stdout(captured):
            console.approval("Tool: edit_file\nAllow?")
            console.reasoning("Applying the approved change.")

        self.assertEqual(
            captured.getvalue(),
            "[approval] Tool: edit_file\nAllow?\n"
            "\n[thinking] Applying the approved change.",
        )

    def test_tty_code_graph_redraws_and_resumes_below_approval(self) -> None:
        state, apply, started, planned, node_id = self._code_events()
        captured = io.StringIO()
        console = cli._Console(verbose=False, color=True)

        with contextlib.redirect_stdout(captured):
            console.code_event(started, state)
            console.code_event(planned, state)
            console.approval("Tool: write_file\nAllow?")
            approved = apply("approval_decided", {"call_key": node_id, "call_id": node_id, "approved": True, "scope": "once"})
            console.code_event(approved, state)
            running = apply("tool_started", {"call_key": node_id, "call_id": node_id, "origin": "code"})
            console.code_event(running, state)

        output = captured.getvalue()
        self.assertIn("\x1b[2K", output)
        self.assertIn("run_code: update service", output)
        self.assertIn("CURRENT", output)
        self.assertLess(output.rfind("[approval]"), output.rfind("CURRENT"))

    def test_non_tty_code_graph_emits_stable_state_lines(self) -> None:
        state, apply, started, planned, node_id = self._code_events()
        captured = io.StringIO()
        console = cli._Console(verbose=False, color=False)

        with contextlib.redirect_stdout(captured):
            console.code_event(started, state)
            console.code_event(planned, state)
            running = apply("tool_started", {"call_key": node_id, "call_id": node_id, "origin": "code"})
            console.code_event(running, state)

        self.assertEqual(
            captured.getvalue().splitlines(),
            [
                "[code-dag] run ACTIVE update service",
                "[code-dag] #1 PLANNED write_file a.txt deps=-",
                "[code-dag] #1 RUNNING write_file a.txt deps=-",
            ],
        )

    def test_non_tty_code_graph_escapes_untrusted_fields(self) -> None:
        state, _, started, planned, _ = self._code_events(
            description="safe\x1b[31m\nforged", path="a\nforged.txt"
        )
        captured = io.StringIO()
        console = cli._Console(verbose=False, color=False)

        with contextlib.redirect_stdout(captured):
            console.code_event(started, state)
            console.code_event(planned, state)

        output = captured.getvalue()
        self.assertNotIn("\x1b", output)
        self.assertIn(r"safe\x1b[31m\nforged", output)
        self.assertIn(r"a\nforged.txt", output)


class CliRunTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.workspace = Path(self.temporary.name).resolve()
        environment = patch.dict(
            os.environ,
            {"MCA_API_KEY": "test-secret-key", "MCA_MODEL": "test-model"},
            clear=True,
        )
        environment.start()
        self.addCleanup(environment.stop)

    def _run(self, argv, model, stdin=""):
        captured = io.StringIO()
        with patch.object(cli, "ModelClient", return_value=model):
            with patch("builtins.input", side_effect=self._input_lines(stdin)):
                with contextlib.redirect_stdout(captured):
                    code = cli.main(
                        [*argv, "--workspace", str(self.workspace)]
                    )
        return code, captured.getvalue()

    def _run_unbound(self, model, stdin="", *, plan=False):
        """Run bare ``mca`` with a test-controlled current directory."""

        captured = io.StringIO()
        argv = ["--plan"] if plan else []
        with patch.object(cli, "ModelClient", return_value=model):
            with patch("builtins.input", side_effect=self._input_lines(stdin)):
                with patch.object(cli.Path, "cwd", return_value=self.workspace):
                    with contextlib.redirect_stdout(captured):
                        code = cli.main(argv)
        return code, captured.getvalue()

    @staticmethod
    def _input_lines(stdin):
        lines = iter(stdin.splitlines())

        def _next(_prompt=""):
            try:
                return next(lines)
            except StopIteration as stop:
                raise EOFError from stop

        return _next

    def test_missing_api_key_reports_error_and_exits_nonzero(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            captured = io.StringIO()
            with contextlib.redirect_stdout(captured):
                code = cli.main(["hello", "--workspace", str(self.workspace)])
        self.assertEqual(code, 1)
        self.assertIn("API key is required", captured.getvalue())

    def test_one_shot_completed_turn_exits_zero(self) -> None:
        code, output = self._run(["say hello"], ScriptedModel(_text("done")))
        self.assertEqual(code, 0)
        self.assertIn("done", output)
        self.assertIn("[session]", output)

    def test_streamed_final_answer_is_printed_once(self) -> None:
        code, output = self._run(
            ["say hello"], StreamingModel(_text("one final answer"))
        )

        self.assertEqual(code, 0)
        self.assertEqual(output.count("one final answer"), 1)

    def test_one_shot_never_leaks_the_api_key(self) -> None:
        _, output = self._run(
            ["--verbose", "say hello"], ScriptedModel(_text("ok"))
        )
        self.assertNotIn("test-secret-key", output)

    def test_yolo_prints_a_visible_warning(self) -> None:
        _, output = self._run(["--yolo", "go"], ScriptedModel(_text("ok")))
        self.assertIn("yolo", output.lower())

    def test_repl_help_and_exit(self) -> None:
        code, output = self._run([], ScriptedModel(), stdin="/help\n/exit\n")
        self.assertEqual(code, 0)
        self.assertIn("/compact", output)
        self.assertIn("/undo", output)

    def test_repl_runs_a_task_then_exits(self) -> None:
        code, output = self._run(
            [], ScriptedModel(_text("first answer")), stdin="do it\n/exit\n"
        )
        self.assertEqual(code, 0)
        self.assertIn("first answer", output)

    def test_resume_rejects_unknown_session(self) -> None:
        code, output = self._run(
            ["--resume", str(uuid.uuid4()), "hi"], ScriptedModel()
        )
        self.assertEqual(code, 1)
        self.assertIn("resume failed", output)

    def test_resume_replays_a_completed_session(self) -> None:
        code, _ = self._run(["remember this"], ScriptedModel(_text("stored")))
        self.assertEqual(code, 0)
        sessions_root = self.workspace / ".mca" / "sessions"
        session_file = next(sessions_root.glob("*.jsonl"))
        session_id = session_file.stem
        with RolloutStore.open(sessions_root, session_id) as store:
            state = SessionReducer.replay(store.load())
        self.assertIsNone(state.active_turn_id)

        code, output = self._run(
            ["--resume", session_id], ScriptedModel(), stdin="/exit\n"
        )
        self.assertEqual(code, 0)
        self.assertIn("resumed session", output)

    def test_list_reports_no_sessions_without_api_key(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            captured = io.StringIO()
            with contextlib.redirect_stdout(captured):
                code = cli.main(["--list", "--workspace", str(self.workspace)])
        self.assertEqual(code, 0)
        self.assertIn("no sessions", captured.getvalue())

    def test_list_summarizes_an_existing_session(self) -> None:
        code, _ = self._run(["remember this"], ScriptedModel(_text("stored")))
        self.assertEqual(code, 0)
        session_id = next(
            (self.workspace / ".mca" / "sessions").glob("*.jsonl")
        ).stem

        code, output = self._run(["--list"], ScriptedModel())
        self.assertEqual(code, 0)
        self.assertIn(session_id, output)
        self.assertIn("last=completed", output)

    def test_show_prints_a_transcript_without_api_key(self) -> None:
        code, _ = self._run(["fix the bug"], ScriptedModel(_text("all done")))
        self.assertEqual(code, 0)
        session_id = next(
            (self.workspace / ".mca" / "sessions").glob("*.jsonl")
        ).stem

        with patch.dict(os.environ, {}, clear=True):
            captured = io.StringIO()
            with contextlib.redirect_stdout(captured):
                code = cli.main(
                    ["--show", session_id, "--workspace", str(self.workspace)]
                )
        self.assertEqual(code, 0)
        transcript = captured.getvalue()
        self.assertIn("fix the bug", transcript)
        self.assertIn("all done", transcript)

    def test_show_graph_expands_a_replayed_code_dag(self) -> None:
        (self.workspace / "seed.txt").write_text("seed", encoding="utf-8")
        code_call = SamplingResult(
            SamplingOutcome.VALID_TOOL_BATCH,
            tool_calls=(
                SampledToolCall(
                    0,
                    "code-1",
                    "function",
                    "run_code",
                    json.dumps(
                        {
                            "description": "inspect seed",
                            "code": (
                                'first = tools.list_dir({"path": "."})\n'
                                'second = tools.read_file({"path": "seed.txt"}, after=[first])\n'
                                "return await second"
                            ),
                        }
                    ),
                ),
            ),
            finish_reason="tool_calls",
        )
        code, _ = self._run(
            ["inspect"], ScriptedModel(code_call, _text("done"))
        )
        self.assertEqual(code, 0)
        session_id = next(
            (self.workspace / ".mca" / "sessions").glob("*.jsonl")
        ).stem

        with patch.dict(os.environ, {}, clear=True):
            captured = io.StringIO()
            with contextlib.redirect_stdout(captured):
                shown = cli.main(
                    ["--show", session_id, "--graph", "--workspace", str(self.workspace)]
                )

        self.assertEqual(shown, 0)
        output = captured.getvalue()
        self.assertIn("run_code: inspect seed SUCCEEDED", output)
        self.assertIn("#1 ──▶ #2", output)

    def test_live_non_tty_run_code_reports_each_dag_transition(self) -> None:
        arguments = json.dumps(
            {
                "description": "inspect workspace",
                "code": 'return await tools.list_dir({"path": "."})',
            }
        )
        code_call = SamplingResult(
            SamplingOutcome.VALID_TOOL_BATCH,
            tool_calls=(
                SampledToolCall(
                    0, "code-1", "function", "run_code", arguments
                ),
            ),
            finish_reason="tool_calls",
        )

        code, output = self._run(
            ["inspect"], ScriptedModel(code_call, _text("done"))
        )

        self.assertEqual(code, 0)
        self.assertIn("[code-dag] run ACTIVE inspect workspace", output)
        self.assertIn("[code-dag] #1 PLANNED list_dir . deps=-", output)
        self.assertIn("[code-dag] #1 RUNNING list_dir . deps=-", output)
        self.assertIn("[code-dag] #1 SUCCEEDED list_dir . deps=-", output)
        self.assertIn("[code-dag] run SUCCEEDED succeeded=1 failed=0 skipped=0", output)

    def test_graph_flag_requires_show(self) -> None:
        with self.assertRaises(SystemExit):
            cli.main(["--graph", "--workspace", str(self.workspace)])

    def test_show_rejects_unknown_session(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            captured = io.StringIO()
            with contextlib.redirect_stdout(captured):
                code = cli.main(
                    [
                        "--show",
                        str(uuid.uuid4()),
                        "--workspace",
                        str(self.workspace),
                    ]
                )
        self.assertEqual(code, 1)
        self.assertIn("does not exist", captured.getvalue())

    def test_repl_status_reports_summary_and_budget(self) -> None:
        code, output = self._run(
            [], ScriptedModel(_text("done")), stdin="do it\n/status\n/exit\n"
        )
        self.assertEqual(code, 0)
        self.assertIn("tokens", output)
        self.assertIn("last=completed", output)

    def test_repl_plan_command_toggles_plan_mode(self) -> None:
        code, output = self._run(
            [], ScriptedModel(), stdin="/plan\n/status\n/plan off\n/exit\n"
        )
        self.assertEqual(code, 0)
        self.assertIn("[plan] plan mode on", output.lower())
        self.assertIn("plan mode: on", output.lower())
        self.assertIn("plan mode off", output.lower())

    def test_repl_approval_reset_is_available(self) -> None:
        code, output = self._run(
            [], ScriptedModel(), stdin="/approval reset\n/exit\n"
        )

        self.assertEqual(code, 0)
        self.assertIn("already prompting for every side effect", output)

    def test_repl_first_prompt_binds_an_explicit_workspace(self) -> None:
        project = self.workspace / "external-project"
        project.mkdir()
        model = ScriptedModel(_text("inspected the target"))

        code, output = self._run_unbound(
            model,
            stdin=(
                f"workspace: {project} | inspect the repository and report its structure\n"
                "/exit\n"
            ),
        )

        self.assertEqual(code, 0)
        self.assertIn("[workspace] bound", output.lower())
        self.assertEqual(len(model._results), 0)
        sessions_root = project / ".mca" / "sessions"
        session_file = next(sessions_root.glob("*.jsonl"))
        self.assertFalse((self.workspace / ".mca" / "sessions").exists())
        with RolloutStore.open(sessions_root, session_file.stem) as store:
            state = SessionReducer.replay(store.load())
        self.assertEqual(state.cwd, str(project.resolve()))
        turn_id = next(iter(state.turn_inputs))
        self.assertEqual(
            state.turn_inputs[turn_id], "inspect the repository and report its structure"
        )

    def test_repl_rejects_workspace_binding_after_a_turn(self) -> None:
        other = self.workspace / "other-project"
        other.mkdir()
        code, output = self._run_unbound(
            ScriptedModel(_text("done")),
            stdin=(
                f"workspace: {self.workspace} | first task\n"
                f"workspace: {other} | second task\n"
                "/exit\n"
            ),
        )

        self.assertEqual(code, 0)
        self.assertIn("only valid before the first turn", output.lower())

    def test_repl_rejects_a_missing_workspace_binding_without_sampling(self) -> None:
        missing = self.workspace / "not-here"
        model = ScriptedModel()
        code, output = self._run_unbound(
            model, stdin=f"workspace: {missing} | investigate\n/exit\n"
        )

        self.assertEqual(code, 0)
        self.assertIn("workspace does not exist", output.lower())
        self.assertEqual(model._results, [])

    def test_plan_flag_starts_in_plan_mode_and_blocks_writes(self) -> None:
        # In plan mode the model asks to write, the runtime refuses without
        # approval, and the second sample gives a text answer.
        write_call = SamplingResult(
            SamplingOutcome.VALID_TOOL_BATCH,
            tool_calls=(
                SampledToolCall(
                    0, "c1", "function", "write_file",
                    '{"path":"x.txt","content":"hi"}',
                ),
            ),
            finish_reason="tool_calls",
        )
        model = ScriptedModel(write_call, _text("here is my plan"))
        code, output = self._run(["--plan", "build a thing"], model)

        self.assertEqual(code, 0)
        self.assertFalse((self.workspace / "x.txt").exists())
        self.assertIn("[plan] plan mode on", output.lower())


if __name__ == "__main__":
    unittest.main()
