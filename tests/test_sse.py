"""Contract tests for incremental SSE decoding and stream assembly."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mca.sse import (
    ProtocolError,
    SSEDecoder,
    StreamAssembler,
    StreamInterruptedError,
)


def event(delta: object, finish_reason: object = None) -> bytes:
    payload = {
        "id": "response-1",
        "choices": [
            {
                "index": 0,
                "delta": delta,
                "finish_reason": finish_reason,
            }
        ],
    }
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n".encode()


DONE = b"data: [DONE]\n\n"


class SSEDecoderTests(unittest.TestCase):
    def test_decodes_utf8_split_across_byte_chunks(self) -> None:
        decoder = SSEDecoder()
        encoded = 'data: {"text":"\u4f60"}\n\n'.encode()
        split = encoded.index("\u4f60".encode()) + 1

        first = decoder.feed(encoded[:split])
        second = decoder.feed(encoded[split:])
        decoder.close()

        self.assertEqual(first, ())
        self.assertEqual(second, ('{"text":"\u4f60"}',))

    def test_ignores_comments_and_blank_events(self) -> None:
        decoder = SSEDecoder()

        decoded = decoder.feed(
            b": keep-alive\r\n\r\n"
            b"event: ignored\r\n"
            b"data: useful\r\n\r\n"
            b"\r\n"
        )

        self.assertEqual(decoded, ("useful",))

    def test_joins_multiple_data_lines_with_newline_and_preserves_done(self) -> None:
        decoder = SSEDecoder()

        decoded = decoder.feed(
            b"data: {\"value\":\n"
            b"data: 1}\n\n"
            b"data: [DONE]\n\n"
        )

        self.assertEqual(decoded, ('{"value":\n1}', "[DONE]"))

    def test_eof_does_not_dispatch_event_without_blank_line(self) -> None:
        decoder = SSEDecoder()

        self.assertEqual(decoder.feed(b"data: incomplete\n"), ())
        self.assertEqual(decoder.close(), ())


class StreamAssemblerTests(unittest.TestCase):
    def test_assembles_text_and_reports_content_deltas(self) -> None:
        displayed: list[str] = []
        assembler = StreamAssembler(on_content=displayed.append)

        assembler.feed(event({"role": "assistant", "content": "Hel"}))
        assembler.feed(event({"content": "lo 你"}))
        assembler.feed(event({}, "stop"))
        assembler.feed(DONE)
        response = assembler.finish()

        self.assertEqual(response.content, "Hello 你")
        self.assertEqual(displayed, ["Hel", "lo 你"])
        self.assertEqual(response.finish_reason, "stop")
        self.assertEqual(response.tool_calls, ())

    def test_assembles_reasoning_content_and_rejects_non_string_delta(self) -> None:
        assembler = StreamAssembler()
        assembler.feed(event({"reasoning_content": "plan "}))
        assembler.feed(event({"reasoning_content": None, "content": "answer"}))
        assembler.feed(event({"reasoning_content": "details"}))
        assembler.feed(event({}, "stop"))
        assembler.feed(DONE)

        response = assembler.finish()

        self.assertEqual(response.reasoning_content, "plan details")
        self.assertEqual(response.content, "answer")

        invalid = StreamAssembler()
        with self.assertRaisesRegex(ProtocolError, "reasoning_content"):
            invalid.feed(event({"reasoning_content": 42}))

    def test_tool_id_and_name_may_arrive_after_arguments_start(self) -> None:
        assembler = StreamAssembler()
        assembler.feed(
            event(
                {
                    "tool_calls": [
                        {
                            "index": 0,
                            "function": {"arguments": '{"path":'},
                        }
                    ]
                }
            )
        )
        assembler.feed(
            event(
                {
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "call-1",
                            "type": "function",
                            "function": {
                                "name": "read_file",
                                "arguments": '"a.py"}',
                            },
                        }
                    ]
                }
            )
        )
        assembler.feed(event({}, "tool_calls"))
        assembler.feed(DONE)

        response = assembler.finish()

        self.assertEqual(len(response.tool_calls), 1)
        call = response.tool_calls[0]
        self.assertEqual(call.index, 0)
        self.assertEqual(call.id, "call-1")
        self.assertEqual(call.type, "function")
        self.assertEqual(call.name, "read_file")
        self.assertEqual(call.arguments, '{"path":"a.py"}')

    def test_interleaved_tool_indexes_keep_independent_argument_order(self) -> None:
        assembler = StreamAssembler()
        deltas = [
            {
                "tool_calls": [
                    {
                        "index": 1,
                        "id": "call-2",
                        "type": "function",
                        "function": {"name": "grep", "arguments": "{"},
                    }
                ]
            },
            {
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "call-1",
                        "type": "function",
                        "function": {"name": "read_file", "arguments": "{"},
                    }
                ]
            },
            {
                "tool_calls": [
                    {
                        "index": 1,
                        "function": {"arguments": '"pattern":"TODO"}'},
                    },
                    {
                        "index": 0,
                        "function": {"arguments": '"path":"a.py"}'},
                    },
                ]
            },
        ]
        for delta in deltas:
            assembler.feed(event(delta))
        assembler.feed(event({}, "tool_calls"))
        assembler.feed(DONE)

        response = assembler.finish()

        self.assertEqual([call.index for call in response.tool_calls], [0, 1])
        self.assertEqual(response.tool_calls[0].arguments, '{"path":"a.py"}')
        self.assertEqual(
            response.tool_calls[1].arguments, '{"pattern":"TODO"}'
        )

    def test_uses_first_choice_and_ignores_additional_choices(self) -> None:
        assembler = StreamAssembler()
        payload = {
            "id": "response-1",
            "choices": [
                {"index": 0, "delta": {"content": "first"}, "finish_reason": None},
                {"index": 1, "delta": {"content": "ignored"}, "finish_reason": "stop"},
            ],
        }
        assembler.feed(f"data: {json.dumps(payload)}\n\n".encode())
        assembler.feed(event({}, "stop"))
        assembler.feed(DONE)

        self.assertEqual(assembler.finish().content, "first")

    def test_repeated_identical_tool_metadata_is_allowed(self) -> None:
        assembler = StreamAssembler()
        metadata = {
            "index": 0,
            "id": "call-1",
            "type": "function",
            "function": {"name": "bash", "arguments": "{"},
        }
        assembler.feed(event({"tool_calls": [metadata]}))
        repeated = dict(metadata)
        repeated["function"] = {"name": "bash", "arguments": "}"}
        assembler.feed(event({"tool_calls": [repeated]}))
        assembler.feed(event({}, "tool_calls"))
        assembler.feed(DONE)

        self.assertEqual(assembler.finish().tool_calls[0].arguments, "{}")

    def test_conflicting_repeated_tool_metadata_is_rejected(self) -> None:
        initial = {
            "index": 0,
            "id": "call-1",
            "type": "function",
            "function": {"name": "bash", "arguments": ""},
        }
        conflicts = (
            {"index": 0, "id": "call-2"},
            {"index": 0, "type": "other"},
            {"index": 0, "function": {"name": "read_file"}},
        )
        for conflict in conflicts:
            with self.subTest(conflict=conflict):
                assembler = StreamAssembler()
                assembler.feed(event({"tool_calls": [initial]}))
                with self.assertRaisesRegex(ProtocolError, "conflicting"):
                    assembler.feed(event({"tool_calls": [conflict]}))

    def test_missing_done_is_an_interrupted_stream(self) -> None:
        assembler = StreamAssembler()
        assembler.feed(event({"content": "partial"}))
        assembler.feed(event({}, "stop"))

        with self.assertRaisesRegex(StreamInterruptedError, "DONE"):
            assembler.finish()

    def test_done_without_finish_reason_is_a_protocol_error(self) -> None:
        assembler = StreamAssembler()
        assembler.feed(event({"content": "partial"}))
        assembler.feed(DONE)

        with self.assertRaisesRegex(ProtocolError, "finish_reason"):
            assembler.finish()

    def test_unknown_finish_reason_is_a_protocol_error(self) -> None:
        assembler = StreamAssembler()
        with self.assertRaisesRegex(ProtocolError, "finish_reason"):
            assembler.feed(event({"content": "text"}, "mystery"))

    def test_missing_critical_choice_fields_are_rejected(self) -> None:
        assembler = StreamAssembler()
        with self.assertRaisesRegex(ProtocolError, "choices"):
            assembler.feed(b'data: {"id":"x","choices":[]}\n\n')

    def test_missing_tool_id_or_name_is_rejected(self) -> None:
        for tool_delta, expected in (
            (
                {
                    "index": 0,
                    "type": "function",
                    "function": {"name": "bash", "arguments": "{}"},
                },
                "id",
            ),
            (
                {
                    "index": 0,
                    "id": "call-1",
                    "type": "function",
                    "function": {"arguments": "{}"},
                },
                "name",
            ),
        ):
            with self.subTest(expected=expected):
                assembler = StreamAssembler()
                assembler.feed(event({"tool_calls": [tool_delta]}))
                assembler.feed(event({}, "tool_calls"))
                assembler.feed(DONE)
                with self.assertRaisesRegex(ProtocolError, expected):
                    assembler.finish()

    def test_malformed_json_is_a_protocol_error(self) -> None:
        assembler = StreamAssembler()

        with self.assertRaisesRegex(ProtocolError, "JSON"):
            assembler.feed(b"data: {not-json}\n\n")


if __name__ == "__main__":
    unittest.main()
