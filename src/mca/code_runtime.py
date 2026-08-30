"""Parent-side lifecycle for the constrained Python Code Mode worker."""

from __future__ import annotations

import os
import queue
import signal
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .code_protocol import DEFAULT_MAX_FRAME_BYTES, ProtocolFrameError, decode_frame, encode_frame


@dataclass(frozen=True)
class CodeRuntimeConfig:
    max_wall_seconds: float = 120.0
    max_source_bytes: int = 64 * 1024
    max_ast_nodes: int = 10_000
    max_eval_steps: int = 100_000
    max_collection_items: int = 10_000
    max_frame_bytes: int = DEFAULT_MAX_FRAME_BYTES
    max_stderr_bytes: int = 16 * 1024

    def __post_init__(self) -> None:
        for name in ("max_source_bytes", "max_ast_nodes", "max_eval_steps", "max_collection_items", "max_frame_bytes", "max_stderr_bytes"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not isinstance(self.max_wall_seconds, (int, float)) or isinstance(self.max_wall_seconds, bool) or self.max_wall_seconds <= 0:
            raise ValueError("max_wall_seconds must be positive")


@dataclass(frozen=True)
class CodeRuntimeError:
    code: str
    message: str


@dataclass(frozen=True)
class CodeRuntimeResult:
    value: Any = None
    logs: tuple[str, ...] = ()
    error: CodeRuntimeError | None = None
    stderr: str = ""


class CodeRuntime:
    _TOOLS = ("read_file", "list_dir", "grep", "write_file", "edit_file", "bash")

    def __init__(self, config: CodeRuntimeConfig | None = None) -> None:
        self.config = config or CodeRuntimeConfig()

    def run(
        self,
        code: str,
        *,
        execute_graph: Callable[[dict[str, object]], dict[str, object]],
    ) -> CodeRuntimeResult:
        if not isinstance(code, str):
            raise TypeError("code must be a string")
        if len(code.encode("utf-8")) > self.config.max_source_bytes:
            return CodeRuntimeResult(error=CodeRuntimeError("SOURCE_LIMIT", "code exceeds source byte limit"))
        worker = Path(__file__).with_name("code_worker.py")
        with tempfile.TemporaryDirectory(prefix="mca-code-") as temporary:
            process = subprocess.Popen(
                [sys.executable, "-I", "-S", "-u", str(worker)],
                cwd=temporary,
                env={},
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True,
            )
            assert process.stdin is not None and process.stdout is not None and process.stderr is not None
            frames: queue.Queue[bytes | BaseException | None] = queue.Queue()
            stderr_chunks: list[bytes] = []

            def read_frames() -> None:
                try:
                    while True:
                        raw = process.stdout.readline(self.config.max_frame_bytes + 1)
                        if not raw:
                            frames.put(None)
                            return
                        frames.put(raw)
                except BaseException as error:
                    frames.put(error)

            def read_stderr() -> None:
                remaining = self.config.max_stderr_bytes
                while remaining > 0:
                    chunk = process.stderr.read(min(4096, remaining))
                    if not chunk:
                        return
                    stderr_chunks.append(chunk)
                    remaining -= len(chunk)

            frame_thread = threading.Thread(target=read_frames, daemon=True)
            error_thread = threading.Thread(target=read_stderr, daemon=True)
            frame_thread.start(); error_thread.start()
            process.stdin.write(encode_frame({
                "type": "start", "code": code, "tools": list(self._TOOLS),
                "max_ast_nodes": self.config.max_ast_nodes,
                "max_eval_steps": self.config.max_eval_steps,
                "max_collection_items": self.config.max_collection_items,
            }, max_bytes=self.config.max_frame_bytes))
            process.stdin.flush()
            result: CodeRuntimeResult | None = None
            deadline = time.monotonic() + self.config.max_wall_seconds
            try:
                while result is None:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        self._stop(process)
                        return CodeRuntimeResult(
                            error=CodeRuntimeError("WALL_TIMEOUT", "code runtime exceeded wall timeout"),
                            stderr=b"".join(stderr_chunks).decode("utf-8", errors="replace"),
                        )
                    try:
                        raw = frames.get(timeout=remaining)
                    except queue.Empty:
                        self._stop(process)
                        return CodeRuntimeResult(
                            error=CodeRuntimeError("WALL_TIMEOUT", "code runtime exceeded wall timeout"),
                            stderr=b"".join(stderr_chunks).decode("utf-8", errors="replace"),
                        )
                    if isinstance(raw, BaseException):
                        raise raw
                    if raw is None:
                        result = CodeRuntimeResult(error=CodeRuntimeError("WORKER_EXIT", "code worker exited before completion"))
                        break
                    try:
                        frame = decode_frame(raw, max_bytes=self.config.max_frame_bytes)
                    except ProtocolFrameError as error:
                        result = CodeRuntimeResult(error=CodeRuntimeError("PROTOCOL_ERROR", str(error)))
                        break
                    if frame.get("type") == "execute_graph":
                        response = execute_graph(frame)
                        process.stdin.write(encode_frame({"type": "graph_result", **response}, max_bytes=self.config.max_frame_bytes))
                        process.stdin.flush()
                        continue
                    if frame.get("type") != "done":
                        result = CodeRuntimeResult(error=CodeRuntimeError("PROTOCOL_ERROR", "unknown worker frame"))
                        break
                    logs = frame.get("logs", [])
                    if not isinstance(logs, list) or any(not isinstance(item, str) for item in logs):
                        result = CodeRuntimeResult(error=CodeRuntimeError("PROTOCOL_ERROR", "worker logs must be strings"))
                    elif isinstance(frame.get("error"), dict):
                        error = frame["error"]
                        result = CodeRuntimeResult(logs=tuple(logs), error=CodeRuntimeError(str(error.get("code", "CODE_ERROR")), str(error.get("message", "code failed"))))
                    else:
                        result = CodeRuntimeResult(value=frame.get("value"), logs=tuple(logs))
            except KeyboardInterrupt:
                self._stop(process, signal.SIGINT)
                raise
            finally:
                if process.poll() is None:
                    self._stop(process)
                try:
                    process.stdin.close()
                except OSError:
                    pass
                frame_thread.join(timeout=1.0); error_thread.join(timeout=1.0)
                for stream in (process.stdout, process.stderr):
                    try:
                        stream.close()
                    except OSError:
                        pass
            assert result is not None
            return CodeRuntimeResult(
                value=result.value, logs=result.logs, error=result.error,
                stderr=b"".join(stderr_chunks).decode("utf-8", errors="replace"),
            )

    @staticmethod
    def _stop(process: subprocess.Popen[bytes], first: signal.Signals = signal.SIGTERM) -> None:
        try:
            os.killpg(process.pid, first)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=0.2)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=1.0)


__all__ = ["CodeRuntime", "CodeRuntimeConfig", "CodeRuntimeError", "CodeRuntimeResult"]
