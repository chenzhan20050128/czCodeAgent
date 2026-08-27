"""Contract tests for the bounded, streaming Chat Completions client."""

from __future__ import annotations

import json
import sys
import unittest
from dataclasses import FrozenInstanceError
from datetime import datetime, timezone
from email.utils import format_datetime
from pathlib import Path
from typing import Callable, Iterator


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mca.model import ModelClient, SampledToolCall

import httpx

from mca.config import Config
from mca.domain import SamplingOutcome


MESSAGES = [{"role": "user", "content": "hello"}]
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file",
            "parameters": {"type": "object", "properties": {}},
        },
    }
]


def choice(delta: object, finish_reason: object = None) -> dict[str, object]:
    return {
        "id": "response-1",
        "object": "chat.completion.chunk",
        "choices": [
            {
                "index": 0,
                "delta": delta,
                "finish_reason": finish_reason,
            }
        ],
    }


def stream_body(*events: dict[str, object], done: bool = True) -> bytes:
    chunks = [
        f"data: {json.dumps(item, separators=(',', ':'))}\n\n"
        for item in events
    ]
    if done:
        chunks.append("data: [DONE]\n\n")
    return "".join(chunks).encode()


def text_body(text: str, finish_reason: str = "stop", *, done: bool = True) -> bytes:
    return stream_body(
        choice({"role": "assistant", "content": text}),
        choice({}, finish_reason),
        done=done,
    )


def response(body: bytes, status: int = 200, **headers: str) -> httpx.Response:
    return httpx.Response(
        status,
        headers={"content-type": "text/event-stream", **headers},
        content=body,
    )


class BrokenStream(httpx.SyncByteStream):
    def __init__(self, first_chunk: bytes) -> None:
        self.first_chunk = first_chunk

    def __iter__(self) -> Iterator[bytes]:
        yield self.first_chunk
        raise httpx.ReadError("stream disconnected")


class FakeTime:
    def __init__(self, now: float = 1_800_000_000.0) -> None:
        self.now = now
        self.sleeps: list[float] = []

    def clock(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


class ModelClientTests(unittest.TestCase):
    def make_client(
        self,
        handler: Callable[[httpx.Request], httpx.Response],
        *,
        max_attempts: int = 3,
        retry_budget_seconds: float = 60.0,
        fake_time: FakeTime | None = None,
        random_value: float = 0.0,
        secret: str = "test-secret",
    ) -> tuple[ModelClient, httpx.Client, FakeTime]:
        timing = fake_time or FakeTime()
        http_client = httpx.Client(transport=httpx.MockTransport(handler))
        config = Config(
            base_url="https://example.test/v1/",
            api_key=secret,
            model="test-model",
            max_attempts=max_attempts,
            retry_budget_seconds=retry_budget_seconds,
        )
        client = ModelClient(
            config,
            client=http_client,
            sleep=timing.sleep,
            clock=timing.clock,
            wall_clock=timing.clock,
            random=lambda: random_value,
        )
        return client, http_client, timing

    def test_returns_complete_text_and_exact_tool_enabled_request(self) -> None:
        requests: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            return response(text_body("hello back"))

        client, http_client, _ = self.make_client(handler)
        self.addCleanup(http_client.close)

        result = client.sample(MESSAGES, TOOLS, allow_tools=True)

        self.assertEqual(result.outcome, SamplingOutcome.COMPLETE_TEXT)
        self.assertEqual(result.content, "hello back")
        self.assertEqual(result.finish_reason, "stop")
        self.assertEqual(result.tool_calls, ())
        self.assertIsNone(result.error)
        self.assertEqual(str(requests[0].url), "https://example.test/v1/chat/completions")
        self.assertEqual(requests[0].headers["authorization"], "Bearer test-secret")
        self.assertEqual(
            json.loads(requests[0].content),
            {
                "model": "test-model",
                "messages": MESSAGES,
                "stream": True,
                "n": 1,
                "tools": TOOLS,
                "tool_choice": "auto",
            },
        )

    def test_finalization_omits_tools_and_forces_tool_choice_none(self) -> None:
        bodies: list[dict[str, object]] = []

        def handler(request: httpx.Request) -> httpx.Response:
            bodies.append(json.loads(request.content))
            return response(text_body("final"))

        client, http_client, _ = self.make_client(handler)
        self.addCleanup(http_client.close)

        result = client.sample(MESSAGES, TOOLS, allow_tools=False)

        self.assertEqual(result.outcome, SamplingOutcome.COMPLETE_TEXT)
        self.assertEqual(
            bodies,
            [
                {
                    "model": "test-model",
                    "messages": MESSAGES,
                    "stream": True,
                    "n": 1,
                    "tool_choice": "none",
                }
            ],
        )

    def test_returns_immutable_tool_batch_even_when_finish_reason_is_stop(self) -> None:
        body = stream_body(
            choice(
                {
                    "content": "I will inspect it.",
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "call-1",
                            "type": "function",
                            "function": {
                                "name": "read_file",
                                "arguments": '{"path":"a.py"}',
                            },
                        }
                    ],
                }
            ),
            choice({}, "stop"),
        )
        client, http_client, _ = self.make_client(lambda _: response(body))
        self.addCleanup(http_client.close)

        result = client.sample(MESSAGES, TOOLS, allow_tools=True)

        self.assertEqual(result.outcome, SamplingOutcome.VALID_TOOL_BATCH)
        self.assertEqual(result.content, "I will inspect it.")
        self.assertEqual(
            result.tool_calls,
            (
                SampledToolCall(
                    index=0,
                    id="call-1",
                    type="function",
                    name="read_file",
                    arguments='{"path":"a.py"}',
                ),
            ),
        )
        with self.assertRaises(FrozenInstanceError):
            result.tool_calls[0].name = "changed"  # type: ignore[misc]

    def test_classifies_empty_stop_length_and_content_filter(self) -> None:
        cases = (
            (text_body(""), SamplingOutcome.PROTOCOL_ERROR),
            (text_body("partial", "length"), SamplingOutcome.LENGTH_EXCEEDED),
            (text_body("blocked", "content_filter"), SamplingOutcome.FILTERED),
        )
        for body, expected in cases:
            with self.subTest(expected=expected):
                client, http_client, _ = self.make_client(
                    lambda _, body=body: response(body)
                )
                self.addCleanup(http_client.close)
                result = client.sample(MESSAGES, TOOLS, allow_tools=True)
                self.assertEqual(result.outcome, expected)

    def test_retries_429_then_succeeds_with_exponential_jitter(self) -> None:
        attempts = 0

        def handler(_: httpx.Request) -> httpx.Response:
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                return httpx.Response(429, json={"error": {"message": "busy"}})
            return response(text_body("recovered"))

        client, http_client, timing = self.make_client(handler, random_value=0.25)
        self.addCleanup(http_client.close)

        result = client.sample(MESSAGES, TOOLS, allow_tools=True)

        self.assertEqual(result.outcome, SamplingOutcome.COMPLETE_TEXT)
        self.assertEqual(result.content, "recovered")
        self.assertEqual(attempts, 2)
        self.assertEqual(timing.sleeps, [1.25])

    def test_honors_retry_after_seconds_and_http_date(self) -> None:
        for retry_after, now, expected in (
            ("3", 1_800_000_000.0, 3.0),
            (
                format_datetime(
                    datetime.fromtimestamp(1_800_000_004.0, timezone.utc),
                    usegmt=True,
                ),
                1_800_000_000.0,
                4.0,
            ),
        ):
            with self.subTest(retry_after=retry_after):
                attempts = 0

                def handler(_: httpx.Request) -> httpx.Response:
                    nonlocal attempts
                    attempts += 1
                    if attempts == 1:
                        return httpx.Response(
                            503, headers={"Retry-After": retry_after}
                        )
                    return response(text_body("ok"))

                timing = FakeTime(now)
                client, http_client, timing = self.make_client(
                    handler, fake_time=timing
                )
                self.addCleanup(http_client.close)
                result = client.sample(MESSAGES, TOOLS, allow_tools=True)
                self.assertEqual(result.outcome, SamplingOutcome.COMPLETE_TEXT)
                self.assertEqual(timing.sleeps, [expected])

    def test_retries_connection_error(self) -> None:
        attempts = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise httpx.ConnectError("cannot connect", request=request)
            return response(text_body("connected"))

        client, http_client, _ = self.make_client(handler)
        self.addCleanup(http_client.close)

        result = client.sample(MESSAGES, TOOLS, allow_tools=True)

        self.assertEqual(result.outcome, SamplingOutcome.COMPLETE_TEXT)
        self.assertEqual(attempts, 2)

    def test_partial_stream_is_invalidated_and_next_attempt_starts_fresh(self) -> None:
        attempts = 0
        displayed: list[str] = []
        invalidations: list[str] = []

        def handler(_: httpx.Request) -> httpx.Response:
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                partial = stream_body(choice({"content": "discard me"}), done=False)
                return httpx.Response(
                    200,
                    headers={"content-type": "text/event-stream"},
                    stream=BrokenStream(partial),
                )
            return response(text_body("fresh answer"))

        client, http_client, _ = self.make_client(handler)
        self.addCleanup(http_client.close)

        result = client.sample(
            MESSAGES,
            TOOLS,
            allow_tools=True,
            on_content=displayed.append,
            on_invalidate=lambda: invalidations.append("invalid"),
        )

        self.assertEqual(result.outcome, SamplingOutcome.COMPLETE_TEXT)
        self.assertEqual(result.content, "fresh answer")
        self.assertEqual(displayed, ["discard me", "fresh answer"])
        self.assertEqual(invalidations, ["invalid"])

    def test_does_not_read_or_retry_after_complete_done_marker(self) -> None:
        attempts = 0

        def handler(_: httpx.Request) -> httpx.Response:
            nonlocal attempts
            attempts += 1
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                stream=BrokenStream(text_body("accepted")),
            )

        client, http_client, timing = self.make_client(handler)
        self.addCleanup(http_client.close)

        result = client.sample(MESSAGES, TOOLS, allow_tools=True)

        self.assertEqual(result.outcome, SamplingOutcome.COMPLETE_TEXT)
        self.assertEqual(result.content, "accepted")
        self.assertEqual(attempts, 1)
        self.assertEqual(timing.sleeps, [])

    def test_retry_attempts_and_budget_are_bounded(self) -> None:
        attempts = 0

        def handler(_: httpx.Request) -> httpx.Response:
            nonlocal attempts
            attempts += 1
            return httpx.Response(503)

        client, http_client, timing = self.make_client(handler, max_attempts=3)
        self.addCleanup(http_client.close)
        result = client.sample(MESSAGES, TOOLS, allow_tools=True)

        self.assertEqual(result.outcome, SamplingOutcome.TRANSPORT_INTERRUPTED)
        self.assertEqual(attempts, 3)
        self.assertEqual(timing.sleeps, [1.0, 2.0])
        self.assertIn("HTTP 503", result.error or "")

        budget_attempts = 0

        def budget_handler(_: httpx.Request) -> httpx.Response:
            nonlocal budget_attempts
            budget_attempts += 1
            return httpx.Response(503)

        budget_client, budget_http_client, budget_time = self.make_client(
            budget_handler, retry_budget_seconds=0.5
        )
        self.addCleanup(budget_http_client.close)
        budget_result = budget_client.sample(MESSAGES, TOOLS, allow_tools=True)

        self.assertEqual(
            budget_result.outcome, SamplingOutcome.TRANSPORT_INTERRUPTED
        )
        self.assertEqual(budget_attempts, 1)
        self.assertEqual(budget_time.sleeps, [])
        self.assertIn("budget", (budget_result.error or "").lower())

    def test_context_overflow_is_not_retried(self) -> None:
        attempts = 0

        def handler(_: httpx.Request) -> httpx.Response:
            nonlocal attempts
            attempts += 1
            return httpx.Response(
                400,
                json={
                    "error": {
                        "code": "context_length_exceeded",
                        "message": "maximum context length exceeded",
                    }
                },
            )

        client, http_client, timing = self.make_client(handler)
        self.addCleanup(http_client.close)

        result = client.sample(MESSAGES, TOOLS, allow_tools=True)

        self.assertEqual(result.outcome, SamplingOutcome.CONTEXT_OVERFLOW)
        self.assertEqual(attempts, 1)
        self.assertEqual(timing.sleeps, [])

    def test_other_client_error_is_protocol_failure_without_secret_leak(self) -> None:
        secret = "unique-key-that-must-never-leak"

        def handler(_: httpx.Request) -> httpx.Response:
            return httpx.Response(
                400,
                json={"error": {"message": f"bad request {secret}"}},
            )

        client, http_client, _ = self.make_client(handler, secret=secret)
        self.addCleanup(http_client.close)

        result = client.sample(MESSAGES, TOOLS, allow_tools=True)
        rendered = repr(result) + repr(client) + str(result.error)

        self.assertEqual(result.outcome, SamplingOutcome.PROTOCOL_ERROR)
        self.assertNotIn(secret, rendered)
        self.assertIn("HTTP 400", result.error or "")


if __name__ == "__main__":
    unittest.main()
