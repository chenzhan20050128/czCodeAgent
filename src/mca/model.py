"""Synchronous OpenAI-compatible model client with bounded retry."""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import timezone
from email.utils import parsedate_to_datetime
from typing import Any

import httpx

from .config import Config
from .domain import SamplingOutcome
from .sse import (
    ProtocolError,
    SampledToolCall,
    StreamAssembler,
    StreamInterruptedError,
)


_RETRYABLE_STATUS_CODES = frozenset({408, 409, 429, 500, 502, 503, 504})
@dataclass(frozen=True)
class SamplingResult:
    """Typed result of one logical sample, including its HTTP attempts."""

    outcome: SamplingOutcome
    content: str = ""
    reasoning_content: str = ""
    tool_calls: tuple[SampledToolCall, ...] = ()
    finish_reason: str | None = None
    error: str | None = None


class ModelClient:
    """Issue streaming Chat Completions requests before any local effects."""

    def __init__(
        self,
        config: Config,
        *,
        client: httpx.Client | None = None,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
        wall_clock: Callable[[], float] = time.time,
        random: Callable[[], float] | None = None,
    ) -> None:
        if client is not None and transport is not None:
            raise ValueError("pass either client or transport, not both")
        if not config.api_key:
            raise ValueError("API key is required for model requests")
        if config.max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        if config.retry_budget_seconds < 0:
            raise ValueError("retry_budget_seconds must be non-negative")

        self._config = config
        self._client = client or httpx.Client(
            transport=transport, timeout=config.request_timeout
        )
        self._owns_client = client is None
        self._sleep = sleep
        self._clock = clock
        self._wall_clock = wall_clock
        self._random = random or __import__("random").random

    def __repr__(self) -> str:
        return (
            f"ModelClient(base_url={self._config.base_url!r}, "
            f"model={self._config.model!r}, max_attempts="
            f"{self._config.max_attempts!r})"
        )

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> ModelClient:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def sample(
        self,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]],
        allow_tools: bool,
        *,
        on_content: Callable[[str], None] | None = None,
        on_invalidate: Callable[[], None] | None = None,
    ) -> SamplingResult:
        """Return one accepted terminal result or a bounded failure."""

        request_body: dict[str, Any] = {
            "model": self._config.model,
            "messages": list(messages),
            "stream": True,
            "n": 1,
            "max_tokens": self._config.max_output_tokens,
        }
        if allow_tools:
            request_body["tools"] = list(tools)

        url = f"{self._config.base_url.rstrip('/')}/chat/completions"
        headers = {"Authorization": f"Bearer {self._config.api_key}"}
        started_at = self._clock()
        last_error = "model request failed"
        content_callback_enabled = True

        for attempt in range(1, self._config.max_attempts + 1):
            remaining = self._config.retry_budget_seconds - (
                self._clock() - started_at
            )
            if remaining <= 0:
                return SamplingResult(
                    SamplingOutcome.TRANSPORT_INTERRUPTED,
                    error=f"{last_error}; retry budget exhausted",
                )
            attempt_timeout = min(self._config.request_timeout, remaining)
            displayed = False

            def display(delta: str) -> None:
                nonlocal displayed, content_callback_enabled
                if delta:
                    displayed = True
                if on_content is not None and content_callback_enabled:
                    try:
                        on_content(delta)
                    except Exception:
                        content_callback_enabled = False

            assembler = StreamAssembler(on_content=display)
            retry_after: str | None = None
            retryable = False
            try:
                with self._client.stream(
                    "POST",
                    url,
                    headers=headers,
                    json=request_body,
                    timeout=httpx.Timeout(attempt_timeout),
                ) as response:
                    if response.status_code != 200:
                        if response.status_code in _RETRYABLE_STATUS_CODES:
                            retryable = True
                            retry_after = response.headers.get("Retry-After")
                            last_error = f"HTTP {response.status_code} from model API"
                        elif response.status_code == 400:
                            try:
                                response.read()
                            except httpx.TransportError:
                                return SamplingResult(
                                    SamplingOutcome.PROTOCOL_ERROR,
                                    error="HTTP 400 from model API",
                                )
                            if _is_context_overflow(response):
                                return SamplingResult(
                                    SamplingOutcome.CONTEXT_OVERFLOW,
                                    error="model context window exceeded",
                                )
                            return SamplingResult(
                                SamplingOutcome.PROTOCOL_ERROR,
                                error="HTTP 400 from model API",
                            )
                        else:
                            return SamplingResult(
                                SamplingOutcome.PROTOCOL_ERROR,
                                error=f"HTTP {response.status_code} from model API",
                            )
                    else:
                        for chunk in response.iter_bytes():
                            assembler.feed(chunk)
                            if assembler.is_done:
                                break
                        candidate = assembler.finish()
                        if candidate.finish_reason == "insufficient_system_resource":
                            retryable = True
                            last_error = "model reported insufficient system resources"
                        else:
                            return _classify(candidate)
            except StreamInterruptedError:
                retryable = True
                last_error = "model stream ended before completion"
            except ProtocolError as exc:
                if displayed:
                    _notify_invalidation(on_invalidate)
                return SamplingResult(
                    SamplingOutcome.PROTOCOL_ERROR, error=_safe_protocol_error(exc)
                )
            except httpx.TransportError as exc:
                retryable = True
                last_error = f"{type(exc).__name__} during model request"

            if not retryable:
                return SamplingResult(
                    SamplingOutcome.PROTOCOL_ERROR, error="model request failed"
                )
            if displayed:
                _notify_invalidation(on_invalidate)
            if attempt >= self._config.max_attempts:
                return SamplingResult(
                    SamplingOutcome.TRANSPORT_INTERRUPTED, error=last_error
                )

            delay = _retry_delay(
                retry_after,
                attempt=attempt,
                wall_time=self._wall_clock(),
                jitter=self._random(),
            )
            remaining = self._config.retry_budget_seconds - (
                self._clock() - started_at
            )
            if remaining <= 0 or delay >= remaining:
                return SamplingResult(
                    SamplingOutcome.TRANSPORT_INTERRUPTED,
                    error=f"{last_error}; retry budget exhausted",
                )
            self._sleep(delay)

        raise AssertionError("unreachable")


def _classify(candidate: Any) -> SamplingResult:
    if candidate.finish_reason == "length":
        return SamplingResult(
            SamplingOutcome.LENGTH_EXCEEDED,
            content=candidate.content,
            reasoning_content=candidate.reasoning_content,
            finish_reason=candidate.finish_reason,
        )
    if candidate.finish_reason == "content_filter":
        return SamplingResult(
            SamplingOutcome.FILTERED,
            content=candidate.content,
            reasoning_content=candidate.reasoning_content,
            finish_reason=candidate.finish_reason,
        )
    outcome = (
        SamplingOutcome.VALID_TOOL_BATCH
        if candidate.tool_calls
        else SamplingOutcome.COMPLETE_TEXT
    )
    return SamplingResult(
        outcome,
        content=candidate.content,
        reasoning_content=candidate.reasoning_content,
        tool_calls=candidate.tool_calls,
        finish_reason=candidate.finish_reason,
    )


def _is_context_overflow(response: httpx.Response) -> bool:
    try:
        document = response.json()
    except (ValueError, TypeError):
        return False
    if not isinstance(document, dict):
        return False
    error = document.get("error")
    if not isinstance(error, dict):
        return False
    code = error.get("code")
    message = error.get("message")
    normalized_code = str(code).lower() if code is not None else ""
    normalized_message = str(message).lower() if message is not None else ""
    return (
        normalized_code
        in {
            "context_length_exceeded",
            "context_window_exceeded",
            "max_tokens_exceeded",
        }
        or "maximum context length" in normalized_message
        or "context length exceeded" in normalized_message
        or "context window" in normalized_message
        and ("exceed" in normalized_message or "too long" in normalized_message)
    )


def _retry_delay(
    retry_after: str | None,
    *,
    attempt: int,
    wall_time: float,
    jitter: float,
) -> float:
    if retry_after is not None:
        value = retry_after.strip()
        try:
            seconds = float(value)
        except ValueError:
            try:
                parsed = parsedate_to_datetime(value)
                if parsed.tzinfo is None:
                    parsed = parsed.replace(tzinfo=timezone.utc)
                seconds = parsed.timestamp() - wall_time
            except (TypeError, ValueError, OverflowError):
                seconds = -1.0
        if seconds >= 0:
            return seconds
    return (2.0 ** (attempt - 1)) + max(0.0, jitter)


def _safe_protocol_error(error: ProtocolError) -> str:
    message = str(error)
    allowed_fragments = (
        "SSE",
        "stream",
        "choice",
        "finish_reason",
        "content",
        "tool call",
        "terminal response",
    )
    if any(fragment in message for fragment in allowed_fragments):
        return message
    return "invalid model stream protocol"


def _notify_invalidation(callback: Callable[[], None] | None) -> None:
    if callback is None:
        return
    try:
        callback()
    except Exception:
        pass


__all__ = ["ModelClient", "SampledToolCall", "SamplingResult"]
