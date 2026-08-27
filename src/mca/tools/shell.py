"""Bounded foreground shell execution with process-group cleanup."""

from __future__ import annotations

import codecs
import os
import signal
import subprocess
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO

from .registry import (
    DEFAULT_MAX_OUTPUT_BYTES,
    DEFAULT_MAX_OUTPUT_LINES,
    TRUNCATION_MARKER,
    ToolResult,
    truncate_output,
)


DEFAULT_TIMEOUT_SECONDS = 120
MAX_TIMEOUT_SECONDS = 600
DEFAULT_TERMINATION_GRACE_SECONDS = 1.0
_MODEL_SECRET_KEYS = frozenset(
    {
        "MCA_API_KEY",
        "DEEPSEEK_API_KEY",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "GEMINI_API_KEY",
        "GOOGLE_API_KEY",
        "AZURE_OPENAI_API_KEY",
        "MISTRAL_API_KEY",
        "GROQ_API_KEY",
    }
)
_STREAM_HEADERS = ("[stdout]\n", "\n[stderr]\n")

OutputCallback = Callable[[str, str], object]


class ShellToolError(RuntimeError):
    """Raised when a shell request cannot be prepared or launched."""


class _CallbackGate:
    """Serialize callback delivery and stop it permanently during cleanup."""

    def __init__(self, callback: OutputCallback | None) -> None:
        self._callback = callback
        self._enabled = True
        self._lock = threading.Lock()

    def emit(self, stream_name: str, text: str) -> None:
        with self._lock:
            callback = self._callback if self._enabled else None
        if callback is not None:
            callback(stream_name, text)

    def disable(self) -> None:
        with self._lock:
            self._enabled = False


@dataclass
class _BoundedCapture:
    """Retain a fixed byte window while the pipe itself is fully drained."""

    max_bytes: int
    _buffer: bytearray = field(default_factory=bytearray)
    _prefix: bytes = b""
    _suffix: bytes = b""
    truncated: bool = False

    def append(self, chunk: bytes) -> None:
        if not chunk:
            return
        if not self.truncated and len(self._buffer) + len(chunk) <= self.max_bytes:
            self._buffer.extend(chunk)
            return

        marker_size = len((f"\n{TRUNCATION_MARKER}\n").encode("utf-8"))
        retained = max(2, self.max_bytes - marker_size)
        head_size = retained // 2
        tail_size = retained - head_size
        if not self.truncated:
            combined = bytes(self._buffer) + chunk
            self._buffer.clear()
            self._prefix = combined[:head_size]
            self._suffix = combined[-tail_size:]
            self.truncated = True
            return
        self._suffix = (self._suffix + chunk)[-tail_size:]

    def text(self) -> str:
        if not self.truncated:
            data = bytes(self._buffer)
        else:
            data = (
                self._prefix
                + f"\n{TRUNCATION_MARKER}\n".encode("utf-8")
                + self._suffix
            )
        return data.decode("utf-8", errors="replace")


@dataclass(frozen=True)
class PreparedShellCommand:
    """A validated command that can be approved before it is started."""

    command: str
    cwd: Path
    timeout_seconds: int
    max_output_bytes: int = field(repr=False)
    max_output_lines: int = field(repr=False)
    termination_grace_seconds: float = field(repr=False)

    def execute(self, *, on_output: OutputCallback | None = None) -> ToolResult:
        environment = os.environ.copy()
        for key in _MODEL_SECRET_KEYS:
            environment.pop(key, None)

        try:
            process = subprocess.Popen(
                ["/bin/sh", "-lc", self.command],
                cwd=self.cwd,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True,
                env=environment,
            )
        except OSError as error:
            raise ShellToolError(f"failed to start shell: {error}") from error

        assert process.stdout is not None
        assert process.stderr is not None
        stdout = _BoundedCapture(self.max_output_bytes)
        stderr = _BoundedCapture(self.max_output_bytes)
        callback_gate = _CallbackGate(on_output)
        drain_stop = threading.Event()
        threads = (
            threading.Thread(
                target=_drain_pipe,
                args=(
                    process.stdout,
                    "stdout",
                    stdout,
                    callback_gate.emit,
                    drain_stop,
                ),
                daemon=True,
            ),
            threading.Thread(
                target=_drain_pipe,
                args=(
                    process.stderr,
                    "stderr",
                    stderr,
                    callback_gate.emit,
                    drain_stop,
                ),
                daemon=True,
            ),
        )
        started_threads: list[threading.Thread] = []
        try:
            for thread in threads:
                thread.start()
                started_threads.append(thread)
        except BaseException as error:
            _stop_process_group(
                process, signal.SIGTERM, self.termination_grace_seconds
            )
            callback_gate.disable()
            drain_stop.set()
            _close_pipes(process.stdout, process.stderr)
            _join_threads(
                tuple(started_threads), timeout=self.termination_grace_seconds
            )
            if isinstance(error, KeyboardInterrupt):
                raise
            raise ShellToolError("failed to start output drain") from error

        timed_out = False
        interrupted = False
        try:
            process.wait(timeout=self.timeout_seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
            _stop_process_group(
                process, signal.SIGTERM, self.termination_grace_seconds
            )
        except KeyboardInterrupt:
            interrupted = True
            _stop_process_group(
                process, signal.SIGINT, self.termination_grace_seconds
            )
        finally:
            if process.poll() is None:
                _stop_process_group(
                    process, signal.SIGTERM, self.termination_grace_seconds
                )
            _wait_process_bounded(process, self.termination_grace_seconds)
            drains_finished = _join_threads(
                tuple(started_threads), timeout=self.termination_grace_seconds
            )
            if not drains_finished:
                timed_out = True
                stderr.append(
                    b"pipe drain did not finish after shell exit; "
                    b"escaped descendants may remain\n"
                )
                _stop_process_group(
                    process, signal.SIGTERM, self.termination_grace_seconds
                )
                callback_gate.disable()
                drain_stop.set()
                _close_pipes(process.stdout, process.stderr)
                drains_finished = _join_threads(
                    tuple(started_threads), timeout=self.termination_grace_seconds
                )
                if not drains_finished:
                    raise ShellToolError(
                        "output drain threads did not stop after bounded cleanup"
                    )
            callback_gate.disable()

        output, rendering_truncated = _render_streams(
            stdout.text(),
            stderr.text(),
            max_bytes=self.max_output_bytes,
            max_lines=self.max_output_lines,
        )
        truncated = stdout.truncated or stderr.truncated or rendering_truncated
        if interrupted:
            status = "interrupted"
        elif timed_out:
            status = "timed_out"
        elif process.returncode == 0:
            status = "succeeded"
        else:
            status = "failed"
        terminating_signal = (
            -process.returncode
            if process.returncode is not None and process.returncode < 0
            else None
        )
        return ToolResult(
            title=f"Bash: {self.command}",
            output=output,
            status=status,
            metadata={
                "exit_code": process.returncode,
                "signal": terminating_signal,
                "timed_out": timed_out,
                "interrupted": interrupted,
                "truncated": truncated,
                "cwd": str(self.cwd),
            },
        )


class ShellRunner:
    """Prepare commands for foreground execution in one fixed workspace."""

    def __init__(
        self,
        workspace: str | os.PathLike[str],
        *,
        default_timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
        max_timeout_seconds: int = MAX_TIMEOUT_SECONDS,
        max_output_bytes: int = DEFAULT_MAX_OUTPUT_BYTES,
        max_output_lines: int = DEFAULT_MAX_OUTPUT_LINES,
        termination_grace_seconds: float = DEFAULT_TERMINATION_GRACE_SECONDS,
    ) -> None:
        cwd = Path(workspace).resolve(strict=True)
        if not cwd.is_dir():
            raise ValueError("workspace must be a directory")
        if type(default_timeout_seconds) is not int or not 1 <= default_timeout_seconds <= max_timeout_seconds:
            raise ValueError("default_timeout_seconds is out of range")
        if type(max_timeout_seconds) is not int or max_timeout_seconds < 1:
            raise ValueError("max_timeout_seconds must be a positive integer")
        if type(max_output_bytes) is not int or max_output_bytes < 64:
            raise ValueError("max_output_bytes must be an integer >= 64")
        if type(max_output_lines) is not int or max_output_lines < 4:
            raise ValueError("max_output_lines must be an integer >= 4")
        if not isinstance(termination_grace_seconds, (int, float)) or isinstance(
            termination_grace_seconds, bool
        ) or termination_grace_seconds <= 0:
            raise ValueError("termination_grace_seconds must be positive")
        self.workspace = cwd
        self.default_timeout_seconds = default_timeout_seconds
        self.max_timeout_seconds = max_timeout_seconds
        self.max_output_bytes = max_output_bytes
        self.max_output_lines = max_output_lines
        self.termination_grace_seconds = float(termination_grace_seconds)

    def prepare(self, arguments: dict[str, Any]) -> PreparedShellCommand:
        if not isinstance(arguments, dict):
            raise ShellToolError("arguments must be an object")
        command = arguments.get("command")
        if not isinstance(command, str) or not command.strip():
            raise ShellToolError("command must be a non-empty string")
        timeout = arguments.get("timeout_seconds", self.default_timeout_seconds)
        if type(timeout) is not int:
            raise ShellToolError("timeout_seconds must be an integer")
        if not 1 <= timeout <= self.max_timeout_seconds:
            raise ShellToolError(
                f"timeout_seconds must be between 1 and {self.max_timeout_seconds}"
            )
        return PreparedShellCommand(
            command=command,
            cwd=self.workspace,
            timeout_seconds=timeout,
            max_output_bytes=self.max_output_bytes,
            max_output_lines=self.max_output_lines,
            termination_grace_seconds=self.termination_grace_seconds,
        )


def _drain_pipe(
    pipe: BinaryIO,
    stream_name: str,
    capture: _BoundedCapture,
    callback: OutputCallback | None,
    stop: threading.Event | None = None,
) -> None:
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    stop = stop or threading.Event()
    try:
        try:
            descriptor = pipe.fileno()
            os.set_blocking(descriptor, False)
        except (OSError, ValueError):
            return
        while not stop.is_set():
            try:
                chunk = os.read(descriptor, 8192)
            except BlockingIOError:
                stop.wait(0.01)
                continue
            except (OSError, ValueError):
                break
            if not chunk:
                break
            capture.append(chunk)
            rendered = decoder.decode(chunk)
            if callback is not None and rendered:
                try:
                    callback(stream_name, rendered)
                except Exception:
                    pass
        rendered = decoder.decode(b"", final=True)
        if callback is not None and rendered:
            try:
                callback(stream_name, rendered)
            except Exception:
                pass
    finally:
        try:
            pipe.close()
        except OSError:
            pass


def _stop_process_group(
    process: subprocess.Popen[bytes], first_signal: signal.Signals, grace: float
) -> None:
    group_signalled = _signal_process_group(process.pid, first_signal)
    if not group_signalled and process.poll() is None:
        try:
            process.send_signal(first_signal)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + grace
    while _process_group_exists(process.pid) and time.monotonic() < deadline:
        if process.poll() is None:
            try:
                process.wait(timeout=min(0.02, max(0.001, deadline - time.monotonic())))
            except subprocess.TimeoutExpired:
                pass
        else:
            time.sleep(min(0.02, max(0.001, deadline - time.monotonic())))
    if _process_group_exists(process.pid):
        group_killed = _signal_process_group(process.pid, signal.SIGKILL)
        if not group_killed and process.poll() is None:
            try:
                process.kill()
            except ProcessLookupError:
                pass
    if process.poll() is None:
        _wait_process_bounded(process, grace)


def _wait_process_bounded(
    process: subprocess.Popen[bytes], timeout: float
) -> bool:
    if process.poll() is not None:
        return True
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        return False
    return True


def _signal_process_group(
    process_group_id: int, requested_signal: signal.Signals
) -> bool:
    try:
        os.killpg(process_group_id, requested_signal)
    except ProcessLookupError:
        return True
    except PermissionError:
        return False
    return True


def _process_group_exists(process_group_id: int) -> bool:
    try:
        os.killpg(process_group_id, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _join_threads(
    threads: tuple[threading.Thread, ...], *, timeout: float
) -> bool:
    deadline = time.monotonic() + timeout
    for thread in threads:
        thread.join(timeout=max(0.0, deadline - time.monotonic()))
    return all(not thread.is_alive() for thread in threads)


def _close_pipes(*pipes: BinaryIO) -> None:
    for pipe in pipes:
        try:
            pipe.close()
        except OSError:
            pass


def _render_streams(
    stdout: str, stderr: str, *, max_bytes: int, max_lines: int
) -> tuple[str, bool]:
    overhead = len((_STREAM_HEADERS[0] + _STREAM_HEADERS[1]).encode("utf-8"))
    content_bytes = max_bytes - overhead
    stdout_bytes = content_bytes // 2
    stderr_bytes = content_bytes - stdout_bytes
    content_lines = max_lines - 2
    stdout_lines = max(1, content_lines // 2)
    stderr_lines = max(1, content_lines - stdout_lines)
    bounded_stdout, stdout_truncated = truncate_output(
        stdout, max_bytes=stdout_bytes, max_lines=stdout_lines
    )
    bounded_stderr, stderr_truncated = truncate_output(
        stderr, max_bytes=stderr_bytes, max_lines=stderr_lines
    )
    output = (
        _STREAM_HEADERS[0]
        + bounded_stdout
        + _STREAM_HEADERS[1]
        + bounded_stderr
    )
    return output, stdout_truncated or stderr_truncated


__all__ = ["PreparedShellCommand", "ShellRunner", "ShellToolError"]
