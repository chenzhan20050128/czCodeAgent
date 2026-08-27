"""Incremental Server-Sent Events decoding for model streams."""

from __future__ import annotations

import codecs
import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


class ProtocolError(ValueError):
    """The provider stream does not match the supported protocol."""


class StreamInterruptedError(ProtocolError):
    """The byte stream ended before the SSE completion marker."""


class SSEDecoder:
    """Decode byte chunks into complete SSE ``data`` event payloads."""

    def __init__(self) -> None:
        self._decoder = codecs.getincrementaldecoder("utf-8")()
        self._text = ""
        self._data_lines: list[str] = []
        self._closed = False

    def feed(self, chunk: bytes) -> tuple[str, ...]:
        if self._closed:
            raise RuntimeError("SSE decoder is closed")
        if not isinstance(chunk, bytes):
            raise TypeError("SSE chunks must be bytes")
        try:
            self._text += self._decoder.decode(chunk, final=False)
        except UnicodeDecodeError as exc:
            raise ProtocolError("SSE stream is not valid UTF-8") from exc
        return self._consume_lines(final=False)

    def close(self) -> tuple[str, ...]:
        if self._closed:
            return ()
        self._closed = True
        try:
            self._text += self._decoder.decode(b"", final=True)
        except UnicodeDecodeError as exc:
            raise ProtocolError("SSE stream ended inside a UTF-8 character") from exc
        return self._consume_lines(final=True)

    def _consume_lines(self, *, final: bool) -> tuple[str, ...]:
        events: list[str] = []
        while True:
            newline_positions = [
                position
                for marker in ("\r", "\n")
                if (position := self._text.find(marker)) >= 0
            ]
            if not newline_positions:
                if final and self._text:
                    self._consume_line(self._text, events)
                    self._text = ""
                break

            position = min(newline_positions)
            marker = self._text[position]
            if marker == "\r" and position + 1 == len(self._text) and not final:
                break
            width = (
                2
                if marker == "\r"
                and position + 1 < len(self._text)
                and self._text[position + 1] == "\n"
                else 1
            )
            line = self._text[:position]
            self._text = self._text[position + width :]
            self._consume_line(line, events)
        return tuple(events)

    def _consume_line(self, line: str, events: list[str]) -> None:
        if line == "":
            if self._data_lines:
                events.append("\n".join(self._data_lines))
                self._data_lines.clear()
            return
        if line.startswith(":"):
            return
        if line == "data":
            self._data_lines.append("")
            return
        if line.startswith("data:"):
            value = line[5:]
            if value.startswith(" "):
                value = value[1:]
            self._data_lines.append(value)


@dataclass(frozen=True)
class SampledToolCall:
    """One fully assembled provider tool call."""

    index: int
    id: str
    type: str
    name: str
    arguments: str


@dataclass(frozen=True)
class StreamResponse:
    """A complete stream candidate, before outcome classification."""

    content: str
    tool_calls: tuple[SampledToolCall, ...]
    finish_reason: str


@dataclass
class _ToolCallParts:
    id: str | None = None
    type: str | None = None
    name: str | None = None
    arguments: str = ""
    saw_arguments: bool = False


_FINISH_REASONS = frozenset({"stop", "tool_calls", "length", "content_filter"})


class StreamAssembler:
    """Assemble the supported Chat Completions streaming subset."""

    def __init__(self, *, on_content: Callable[[str], None] | None = None) -> None:
        self._decoder = SSEDecoder()
        self._on_content = on_content
        self._content: list[str] = []
        self._tool_calls: dict[int, _ToolCallParts] = {}
        self._finish_reason: str | None = None
        self._done = False
        self._finished = False

    def feed(self, chunk: bytes) -> None:
        if self._finished:
            raise RuntimeError("stream assembler is finished")
        for payload in self._decoder.feed(chunk):
            self._consume_payload(payload)

    @property
    def is_done(self) -> bool:
        """Whether the exact SSE completion marker has been received."""

        return self._done

    def finish(self) -> StreamResponse:
        if self._finished:
            raise RuntimeError("stream assembler is finished")
        self._finished = True
        for payload in self._decoder.close():
            self._consume_payload(payload)

        if not self._done:
            raise StreamInterruptedError("SSE stream ended before [DONE]")
        if self._finish_reason is None:
            raise ProtocolError("stream is missing a terminal finish_reason")

        calls: list[SampledToolCall] = []
        for index in sorted(self._tool_calls):
            parts = self._tool_calls[index]
            if not parts.id:
                raise ProtocolError(f"tool call {index} is missing id")
            if parts.type != "function":
                raise ProtocolError(f"tool call {index} type must be function")
            if not parts.name:
                raise ProtocolError(f"tool call {index} is missing function name")
            if not parts.saw_arguments:
                raise ProtocolError(f"tool call {index} is missing function arguments")
            calls.append(
                SampledToolCall(
                    index=index,
                    id=parts.id,
                    type=parts.type,
                    name=parts.name,
                    arguments=parts.arguments,
                )
            )

        content = "".join(self._content)
        if self._finish_reason in {"stop", "tool_calls"} and not content and not calls:
            raise ProtocolError("terminal response is empty")
        if self._finish_reason == "tool_calls" and not calls:
            raise ProtocolError("tool_calls finish_reason has no tool calls")
        return StreamResponse(content, tuple(calls), self._finish_reason)

    def _consume_payload(self, payload: str) -> None:
        if payload == "[DONE]":
            if self._done:
                raise ProtocolError("duplicate [DONE] marker")
            self._done = True
            return
        if self._done:
            raise ProtocolError("received data after [DONE]")
        if self._finish_reason is not None:
            raise ProtocolError("received data after terminal finish_reason")

        try:
            document = json.loads(payload)
        except (json.JSONDecodeError, TypeError):
            raise ProtocolError("SSE data is not valid JSON") from None
        if not isinstance(document, dict):
            raise ProtocolError("stream event must be a JSON object")
        choices = document.get("choices")
        if not isinstance(choices, list) or len(choices) != 1:
            raise ProtocolError("stream event choices must contain exactly one item")
        choice = choices[0]
        if not isinstance(choice, dict):
            raise ProtocolError("stream choice must be an object")
        if type(choice.get("index")) is not int or choice["index"] != 0:
            raise ProtocolError("stream choice index must be 0")
        if "finish_reason" not in choice:
            raise ProtocolError("stream choice is missing finish_reason")
        finish_reason = choice["finish_reason"]
        if finish_reason is not None and finish_reason not in _FINISH_REASONS:
            raise ProtocolError("unsupported finish_reason")
        delta = choice.get("delta")
        if not isinstance(delta, dict):
            raise ProtocolError("stream choice delta must be an object")

        self._consume_delta(delta)
        if finish_reason is not None:
            self._finish_reason = finish_reason

    def _consume_delta(self, delta: dict[str, Any]) -> None:
        if "content" in delta and delta["content"] is not None:
            content = delta["content"]
            if not isinstance(content, str):
                raise ProtocolError("content delta must be a string or null")
            self._content.append(content)
            if self._on_content is not None:
                self._on_content(content)

        if "tool_calls" not in delta:
            return
        tool_deltas = delta["tool_calls"]
        if not isinstance(tool_deltas, list):
            raise ProtocolError("tool_calls delta must be a list")
        for tool_delta in tool_deltas:
            self._consume_tool_delta(tool_delta)

    def _consume_tool_delta(self, tool_delta: object) -> None:
        if not isinstance(tool_delta, dict):
            raise ProtocolError("tool call delta must be an object")
        index = tool_delta.get("index")
        if type(index) is not int or index < 0:
            raise ProtocolError("tool call index must be a non-negative integer")
        parts = self._tool_calls.setdefault(index, _ToolCallParts())

        if "id" in tool_delta:
            if not isinstance(tool_delta["id"], str):
                raise ProtocolError("tool call id must be a string")
            parts.id = tool_delta["id"]
        if "type" in tool_delta:
            if not isinstance(tool_delta["type"], str):
                raise ProtocolError("tool call type must be a string")
            parts.type = tool_delta["type"]
        if "function" not in tool_delta:
            return
        function = tool_delta["function"]
        if not isinstance(function, dict):
            raise ProtocolError("tool call function must be an object")
        if "name" in function:
            if not isinstance(function["name"], str):
                raise ProtocolError("tool call function name must be a string")
            parts.name = function["name"]
        if "arguments" in function:
            arguments = function["arguments"]
            if not isinstance(arguments, str):
                raise ProtocolError("tool call arguments must be a string")
            parts.saw_arguments = True
            parts.arguments += arguments


__all__ = [
    "ProtocolError",
    "SSEDecoder",
    "SampledToolCall",
    "StreamAssembler",
    "StreamInterruptedError",
    "StreamResponse",
]
