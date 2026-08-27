"""Bounded foreground shell execution with process-group cleanup."""

from __future__ import annotations

import codecs
import os
import signal
import subprocess
import threading
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
        threads = (
            threading.Thread(
                target=_drain_pipe,
                args=(process.stdout, "stdout", stdout, on_output),
                daemon=True,
            ),
            threading.Thread(
                target=_drain_pipe,
                args=(process.stderr, "stderr", stderr, on_output),
                daemon=True,
            ),
        )
        for thread in threads:
            thread.start()

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
            process.wait()
            for thread in threads:
                thread.join()

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
) -> None:
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    try:
        while True:
            chunk = os.read(pipe.fileno(), 8192)
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
        pipe.close()


def _stop_process_group(
    process: subprocess.Popen[bytes], first_signal: signal.Signals, grace: float
) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, first_signal)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=grace)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()


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
