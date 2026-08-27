"""Real shell execution, output bounds, and process cleanup tests."""

from __future__ import annotations

import _thread
import os
import shlex
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mca.tools import create_tool_registry
from mca.tools.registry import ToolValidationError
from mca.tools.shell import (
    BoundedOutputChannel,
    ShellRunner,
    ShellToolError,
    _BoundedCapture,
    _drain_pipe,
    _stop_process_group,
)


class ShellRunnerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.workspace = Path(self.temporary.name).resolve()

    def python_command(self, source: str) -> str:
        return f"{shlex.quote(sys.executable)} -c {shlex.quote(source)}"

    def test_prepare_requires_nonempty_command_and_bounded_integer_timeout(self) -> None:
        runner = ShellRunner(self.workspace)
        invalid = (
            ({"command": ""}, "command must be a non-empty string"),
            ({"command": "   "}, "command must be a non-empty string"),
            ({"command": "pwd", "timeout_seconds": True}, "integer"),
            ({"command": "pwd", "timeout_seconds": 1.5}, "integer"),
            ({"command": "pwd", "timeout_seconds": 0}, "between"),
            ({"command": "pwd", "timeout_seconds": 601}, "between"),
        )
        for arguments, message in invalid:
            with self.subTest(arguments=arguments):
                with self.assertRaisesRegex(ShellToolError, message):
                    runner.prepare(arguments)

        prepared = runner.prepare({"command": "pwd", "timeout_seconds": 5})
        self.assertEqual(prepared.command, "pwd")
        self.assertEqual(prepared.cwd, self.workspace)
        self.assertEqual(prepared.timeout_seconds, 5)

    def test_registry_exposes_integer_timeout_and_rejects_float(self) -> None:
        registry = create_tool_registry(self.workspace)
        schema = registry.resolve("bash").schema

        self.assertEqual(schema["properties"]["command"]["minLength"], 1)
        self.assertEqual(schema["properties"]["timeout_seconds"]["type"], "integer")
        with self.assertRaisesRegex(ToolValidationError, "must not be empty"):
            registry.parse_and_validate("bash", '{"command":""}')
        with self.assertRaisesRegex(ToolValidationError, "integer"):
            registry.parse_and_validate(
                "bash", '{"command":"pwd","timeout_seconds":1.5}'
            )

    def test_uses_fixed_shell_cwd_devnull_new_session_and_records_nonzero(self) -> None:
        runner = ShellRunner(self.workspace)
        command = "pwd; printf out; printf err >&2; exit 7"
        real_popen = subprocess.Popen

        with patch("mca.tools.shell.subprocess.Popen", wraps=real_popen) as popen:
            result = runner.prepare({"command": command}).execute()

        args, kwargs = popen.call_args
        self.assertEqual(args[0], ["/bin/sh", "-lc", command])
        self.assertEqual(Path(kwargs["cwd"]), self.workspace)
        self.assertIs(kwargs["stdin"], subprocess.DEVNULL)
        self.assertIs(kwargs["stdout"], subprocess.PIPE)
        self.assertIs(kwargs["stderr"], subprocess.PIPE)
        self.assertIs(kwargs["start_new_session"], True)
        self.assertEqual(result.status, "failed")
        self.assertEqual(result.metadata["exit_code"], 7)
        self.assertIs(result.metadata["timed_out"], False)
        self.assertIs(result.metadata["interrupted"], False)
        self.assertIn(f"[stdout]\n{self.workspace}\nout", result.output)
        self.assertIn("[stderr]\nerr", result.output)

    def test_stdin_is_devnull(self) -> None:
        command = self.python_command(
            "import sys; print('stdin-empty' if sys.stdin.read() == '' else 'stdin-data')"
        )
        result = ShellRunner(self.workspace).prepare({"command": command}).execute()

        self.assertEqual(result.status, "succeeded")
        self.assertIn("stdin-empty", result.output)

    def test_large_stdout_and_stderr_are_drained_without_deadlock_and_bounded(self) -> None:
        command = self.python_command(
            "import os; "
            "[(os.write(1, b'O' * 8192), os.write(2, b'E' * 8192)) "
            "for _ in range(80)]"
        )
        runner = ShellRunner(
            self.workspace, max_output_bytes=4_096, max_output_lines=40
        )

        result = runner.prepare({"command": command, "timeout_seconds": 10}).execute()

        self.assertEqual(result.status, "succeeded")
        self.assertEqual(result.metadata["exit_code"], 0)
        self.assertIs(result.metadata["truncated"], True)
        self.assertLessEqual(len(result.output.encode("utf-8")), 4_096)
        self.assertIn("[stdout]", result.output)
        self.assertIn("[stderr]", result.output)

    def test_output_channel_receives_stdout_and_stderr(self) -> None:
        channel = BoundedOutputChannel(capacity=8)

        result = ShellRunner(self.workspace).prepare(
            {"command": "printf hello; printf world >&2"}
        ).execute(output_channel=channel)

        chunks = channel.drain()
        self.assertEqual(result.status, "succeeded")
        self.assertEqual({stream for stream, _ in chunks}, {"stdout", "stderr"})
        self.assertEqual(
            "".join(text for stream, text in chunks if stream == "stdout"),
            "hello",
        )
        self.assertEqual(
            "".join(text for stream, text in chunks if stream == "stderr"),
            "world",
        )

    def test_output_channel_has_fixed_nonblocking_capacity(self) -> None:
        channel = BoundedOutputChannel(capacity=1)

        self.assertIs(channel.offer("stdout", "first"), True)
        self.assertIs(channel.offer("stderr", "dropped"), False)
        self.assertEqual(channel.poll(), ("stdout", "first"))
        self.assertIsNone(channel.poll())

    def test_full_unconsumed_output_channel_does_not_block_command(self) -> None:
        channel = BoundedOutputChannel(capacity=1)
        command = self.python_command(
            "import os; "
            "[(os.write(1, b'O' * 8192), os.write(2, b'E' * 8192)) "
            "for _ in range(80)]"
        )
        started = time.monotonic()

        result = ShellRunner(self.workspace).prepare(
            {"command": command, "timeout_seconds": 10}
        ).execute(output_channel=channel)

        self.assertLess(time.monotonic() - started, 3)
        self.assertEqual(result.status, "succeeded")
        self.assertEqual(len(channel.drain()), 1)
        self.assertIn("O", result.output)
        self.assertIn("E", result.output)

    def test_full_unconsumed_channel_preserves_timeout_and_drain_cleanup(self) -> None:
        channel = BoundedOutputChannel(capacity=1)
        real_thread = threading.Thread
        created_threads: list[tuple[object, threading.Thread]] = []
        thread_errors: list[BaseException] = []
        original_excepthook = threading.excepthook

        def recording_thread(*args: object, **kwargs: object) -> threading.Thread:
            thread = real_thread(*args, **kwargs)
            created_threads.append((kwargs.get("target"), thread))
            return thread

        threading.excepthook = lambda args: thread_errors.append(args.exc_value)
        try:
            started = time.monotonic()
            with patch(
                "mca.tools.shell.threading.Thread",
                side_effect=recording_thread,
            ):
                result = ShellRunner(
                    self.workspace, termination_grace_seconds=0.1
                ).prepare(
                    {
                        "command": "printf ready; printf error >&2; sleep 60",
                        "timeout_seconds": 1,
                    }
                ).execute(output_channel=channel)
        finally:
            threading.excepthook = original_excepthook

        self.assertLess(time.monotonic() - started, 3)
        self.assertEqual(result.status, "timed_out")
        drain_threads = [
            thread
            for target, thread in created_threads
            if getattr(target, "__name__", None) == "_drain_pipe"
        ]
        self.assertEqual(len(created_threads), 2)
        self.assertEqual(len(drain_threads), 2)
        self.assertTrue(all(not thread.is_alive() for thread in drain_threads))
        self.assertEqual(thread_errors, [])

    def test_shell_execute_rejects_callable_output_channel(self) -> None:
        prepared = ShellRunner(self.workspace).prepare({"command": "printf no"})

        with self.assertRaisesRegex(TypeError, "BoundedOutputChannel"):
            prepared.execute(output_channel=lambda stream, text: None)
        with self.assertRaisesRegex(TypeError, "unexpected keyword argument"):
            prepared.execute(on_output=lambda stream, text: None)

    def test_known_model_secrets_are_removed_but_ordinary_environment_remains(self) -> None:
        keys = (
            "MCA_API_KEY",
            "DEEPSEEK_API_KEY",
            "OPENAI_API_KEY",
            "ANTHROPIC_API_KEY",
            "GEMINI_API_KEY",
        )
        source = (
            "import os; "
            f"print(','.join('1' if os.getenv(k) else '0' for k in {keys!r})); "
            "print(os.getenv('MCA_TEST_VISIBLE', 'missing'))"
        )
        with patch.dict(
            os.environ,
            {**{key: "super-secret" for key in keys}, "MCA_TEST_VISIBLE": "kept"},
            clear=False,
        ):
            result = ShellRunner(self.workspace).prepare(
                {"command": self.python_command(source)}
            ).execute()

        self.assertIn("0,0,0,0,0\nkept", result.output)
        self.assertNotIn("super-secret", result.output)

    def test_timeout_terminates_process_group_and_reaps_child(self) -> None:
        command = (
            "sleep 60 & child=$!; "
            "printf '%s' \"$child\" > child.pid; "
            "wait \"$child\""
        )
        result = ShellRunner(self.workspace, termination_grace_seconds=0.1).prepare(
            {"command": command, "timeout_seconds": 1}
        ).execute()

        child_pid = int((self.workspace / "child.pid").read_text(encoding="utf-8"))
        self.assertEqual(result.status, "timed_out")
        self.assertIs(result.metadata["timed_out"], True)
        self.assertIs(result.metadata["interrupted"], False)
        self.assertIsNotNone(result.metadata["exit_code"])
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            try:
                os.kill(child_pid, 0)
            except ProcessLookupError:
                break
            time.sleep(0.02)
        else:
            self.fail(f"child process {child_pid} survived timeout cleanup")

    def test_parent_exit_with_descendant_holding_pipes_is_bounded_and_cleaned(self) -> None:
        command = (
            "(trap '' TERM; sleep 60) & child=$!; "
            "printf '%s' \"$child\" > detached.pid; "
            "exit 0"
        )
        outcome: list[object] = []

        def execute() -> None:
            outcome.append(
                ShellRunner(self.workspace, termination_grace_seconds=0.1)
                .prepare({"command": command, "timeout_seconds": 1})
                .execute()
            )

        thread_errors: list[BaseException] = []
        original_excepthook = threading.excepthook
        threading.excepthook = lambda args: thread_errors.append(args.exc_value)
        worker = threading.Thread(target=execute, daemon=True)
        try:
            worker.start()
            worker.join(timeout=2)
        finally:
            threading.excepthook = original_excepthook
        child_pid = int((self.workspace / "detached.pid").read_text(encoding="utf-8"))
        if worker.is_alive():
            os.kill(child_pid, signal.SIGKILL)
            worker.join(timeout=2)
            self.fail("shell execution hung while a descendant held its pipes")
        result = outcome[0]
        self.assertEqual(result.status, "timed_out")
        self.assertIs(result.metadata["timed_out"], True)
        self.assertIn("pipe drain", result.output)
        self.assertIn("escaped descendants may remain", result.output)
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            try:
                os.kill(child_pid, 0)
            except ProcessLookupError:
                break
            time.sleep(0.02)
        else:
            self.fail(f"descendant process {child_pid} survived pipe cleanup")
        self.assertEqual(thread_errors, [])

    def test_escaped_setsid_descendant_cannot_leave_drain_threads_alive(self) -> None:
        command = (
            self.python_command(
                "import os,time; "
                "os.setsid(); "
                "pid_file=open('escaped.pid','w'); "
                "pid_file.write(str(os.getpid())); pid_file.close(); "
                "time.sleep(60)"
            )
            + " & exit 0"
        )
        real_thread = threading.Thread
        real_popen = subprocess.Popen
        created_threads: list[tuple[object, threading.Thread]] = []
        processes: list[subprocess.Popen[bytes]] = []
        thread_errors: list[BaseException] = []
        original_excepthook = threading.excepthook
        escaped_pid: int | None = None

        def recording_thread(*args: object, **kwargs: object) -> threading.Thread:
            thread = real_thread(*args, **kwargs)
            created_threads.append((kwargs.get("target"), thread))
            return thread

        def recording_popen(
            *args: object, **kwargs: object
        ) -> subprocess.Popen[bytes]:
            process = real_popen(*args, **kwargs)
            processes.append(process)
            return process

        threading.excepthook = lambda args: thread_errors.append(args.exc_value)
        try:
            with (
                patch(
                    "mca.tools.shell.threading.Thread", side_effect=recording_thread
                ),
                patch("mca.tools.shell.subprocess.Popen", side_effect=recording_popen),
                patch("mca.tools.shell._close_pipes"),
            ):
                result = ShellRunner(
                    self.workspace, termination_grace_seconds=0.1
                ).prepare(
                    {"command": command, "timeout_seconds": 2}
                ).execute()

            pid_path = self.workspace / "escaped.pid"
            self.assertTrue(pid_path.exists())
            escaped_pid = int(pid_path.read_text(encoding="utf-8"))
            self.assertEqual(result.status, "timed_out")
            self.assertIn("escaped descendants may remain", result.output)
            drain_threads = [
                thread
                for target, thread in created_threads
                if getattr(target, "__name__", None) == "_drain_pipe"
            ]
            self.assertEqual(len(drain_threads), 2)
            self.assertTrue(all(not thread.is_alive() for thread in drain_threads))
            self.assertEqual(thread_errors, [])
        finally:
            if escaped_pid is None:
                pid_path = self.workspace / "escaped.pid"
                if pid_path.exists():
                    escaped_pid = int(pid_path.read_text(encoding="utf-8"))
            if escaped_pid is not None:
                try:
                    os.kill(escaped_pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            for process in processes:
                if process.stdout is not None:
                    process.stdout.close()
                if process.stderr is not None:
                    process.stderr.close()
            for _, thread in created_threads:
                thread.join(timeout=1)
            threading.excepthook = original_excepthook

    def test_drain_pipe_closed_before_fileno_has_no_background_exception(self) -> None:
        read_fd, write_fd = os.pipe()
        pipe = os.fdopen(read_fd, "rb", buffering=0)
        pipe.close()
        os.close(write_fd)
        thread_errors: list[BaseException] = []
        original_excepthook = threading.excepthook
        threading.excepthook = lambda args: thread_errors.append(args.exc_value)
        worker = threading.Thread(
            target=_drain_pipe,
            args=(pipe, "stdout", _BoundedCapture(64), None),
            daemon=True,
        )
        try:
            worker.start()
            worker.join(timeout=1)
        finally:
            threading.excepthook = original_excepthook

        self.assertFalse(worker.is_alive())
        self.assertEqual(thread_errors, [])

    def test_group_signal_permission_error_falls_back_to_parent_cleanup(self) -> None:
        started = time.monotonic()
        with patch("mca.tools.shell.os.killpg", side_effect=PermissionError):
            result = ShellRunner(
                self.workspace, termination_grace_seconds=0.05
            ).prepare(
                {"command": "sleep 60", "timeout_seconds": 1}
            ).execute()

        self.assertLess(time.monotonic() - started, 3)
        self.assertEqual(result.status, "timed_out")
        self.assertIsNotNone(result.metadata["exit_code"])

    def test_thread_start_failure_reaps_started_shell(self) -> None:
        real_popen = subprocess.Popen
        processes: list[subprocess.Popen[bytes]] = []

        def recording_popen(*args: object, **kwargs: object) -> subprocess.Popen[bytes]:
            process = real_popen(*args, **kwargs)
            processes.append(process)
            return process

        with (
            patch("mca.tools.shell.subprocess.Popen", side_effect=recording_popen),
            patch(
                "mca.tools.shell.threading.Thread.start",
                side_effect=RuntimeError("cannot start drain"),
            ),
        ):
            with self.assertRaisesRegex(ShellToolError, "output drain"):
                ShellRunner(
                    self.workspace, termination_grace_seconds=0.05
                ).prepare(
                    {"command": "sleep 60", "timeout_seconds": 10}
                ).execute()

        self.assertEqual(len(processes), 1)
        self.assertIsNotNone(processes[0].poll())

    def test_keyboard_interrupt_during_thread_start_reaps_then_propagates(self) -> None:
        real_popen = subprocess.Popen
        processes: list[subprocess.Popen[bytes]] = []

        def recording_popen(*args: object, **kwargs: object) -> subprocess.Popen[bytes]:
            process = real_popen(*args, **kwargs)
            processes.append(process)
            return process

        with (
            patch("mca.tools.shell.subprocess.Popen", side_effect=recording_popen),
            patch(
                "mca.tools.shell.threading.Thread.start",
                side_effect=KeyboardInterrupt,
            ),
        ):
            with self.assertRaises(KeyboardInterrupt):
                ShellRunner(
                    self.workspace, termination_grace_seconds=0.05
                ).prepare(
                    {"command": "sleep 60", "timeout_seconds": 10}
                ).execute()

        self.assertEqual(len(processes), 1)
        self.assertIsNotNone(processes[0].poll())

    def test_sigkill_cleanup_never_uses_unbounded_wait(self) -> None:
        process = Mock()
        process.pid = 12345
        process.poll.return_value = None
        process.wait.side_effect = subprocess.TimeoutExpired("test", 0.001)

        with (
            patch("mca.tools.shell._signal_process_group", return_value=True),
            patch("mca.tools.shell._process_group_exists", return_value=True),
        ):
            _stop_process_group(process, signal.SIGTERM, 0.001)

        self.assertGreaterEqual(process.wait.call_count, 1)
        self.assertTrue(
            all(call.kwargs.get("timeout") is not None for call in process.wait.call_args_list)
        )

    def test_keyboard_interrupt_interrupts_group_and_returns_terminal_result(self) -> None:
        runner = ShellRunner(self.workspace, termination_grace_seconds=0.1)
        timer = threading.Timer(0.2, _thread.interrupt_main)
        timer.start()
        try:
            result = runner.prepare(
                {"command": "sleep 60", "timeout_seconds": 10}
            ).execute()
        finally:
            timer.cancel()
            timer.join()

        self.assertEqual(result.status, "interrupted")
        self.assertIs(result.metadata["interrupted"], True)
        self.assertIs(result.metadata["timed_out"], False)
        self.assertIsNotNone(result.metadata["exit_code"])


if __name__ == "__main__":
    unittest.main()
