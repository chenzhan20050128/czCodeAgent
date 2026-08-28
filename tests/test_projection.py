"""Prompt projection and request-budget contract tests."""

from __future__ import annotations

import json
import sys
import unittest
import uuid
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mca.domain import DomainError, Event, SessionReducer, SessionState
from mca.projection import (
    ProjectionBlockedError,
    ProjectionEnvironment,
    ProjectionError,
    PromptProjector,
    estimate_request_tokens,
    request_fits_budget,
    validate_conversation,
)
from mca.sse import StreamAssembler


class ProjectionTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.session_id = str(uuid.uuid4())
        self.turn_id = str(uuid.uuid4())
        self.state = SessionState()
        self.events: list[Event] = []
        self.environment = ProjectionEnvironment(
            cwd="/live/worktree",
            platform="linux",
            date="2026-08-27",
            is_git=True,
        )
        self.system_message = {
            "role": "system",
            "content": (
                "You are a coding assistant. You propose tool calls; the local "
                "MCA runtime validates and executes them. Do not claim that you "
                "executed a tool yourself. Never reveal or persist secrets.\n"
                "Current live environment (supplied for this request, not "
                "recovered from the rollout): "
                '{"cwd":"/live/worktree","date":"2026-08-27",'
                '"is_git":true,"platform":"linux"}'
            ),
        }

    def apply(self, event_type: str, payload: dict[str, object]) -> Event:
        event = Event.create(
            seq=len(self.events) + 1,
            session_id=self.session_id,
            event_type=event_type,
            payload=payload,
        )
        SessionReducer.apply(self.state, event)
        self.events.append(event)
        return event

    def create_session(self) -> None:
        self.apply(
            "session_created",
            {
                "cwd": "/stale/rollout/cwd",
                "model": "test-model",
                "context_window": 4096,
            },
        )

    def start_turn(self, user_input: str = "fix the bug") -> None:
        self.create_session()
        self.apply(
            "turn_started",
            {"turn_id": self.turn_id, "user_input": user_input},
        )

    @staticmethod
    def tool_call(
        call_id: str, name: str, arguments: str
    ) -> dict[str, object]:
        return {
            "id": call_id,
            "type": "function",
            "function": {"name": name, "arguments": arguments},
        }

    def project(self) -> list[dict[str, object]]:
        return PromptProjector.project(
            self.events, self.state, self.environment
        )


class PromptProjectionTests(ProjectionTestCase):
    def test_preserves_reasoning_content_for_followup_after_tool_result(self) -> None:
        self.start_turn("Inspect the file")
        assembler = StreamAssembler()
        streamed = {
            "id": "response-1",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "reasoning_content": "I need the file contents.",
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
                    },
                    "finish_reason": None,
                }
            ],
        }
        terminal = {
            "id": "response-1",
            "choices": [
                {"index": 0, "delta": {}, "finish_reason": "tool_calls"}
            ],
        }
        for item in (streamed, terminal):
            assembler.feed(f"data: {json.dumps(item)}\n\n".encode())
        assembler.feed(b"data: [DONE]\n\n")
        sampled = assembler.finish()
        call = {
            "id": sampled.tool_calls[0].id,
            "type": sampled.tool_calls[0].type,
            "function": {
                "name": sampled.tool_calls[0].name,
                "arguments": sampled.tool_calls[0].arguments,
            },
        }
        self.apply(
            "assistant_accepted",
            {
                "content": sampled.content or None,
                "reasoning_content": sampled.reasoning_content,
                "finish_reason": sampled.finish_reason,
                "tool_calls": [call],
            },
        )
        self.apply(
            "tool_finished",
            {
                "call_id": "call-1",
                "status": "invalid_arguments",
                "result": "contents",
            },
        )

        messages = self.project()

        self.assertEqual(
            messages[-2],
            {
                "role": "assistant",
                "content": None,
                "reasoning_content": "I need the file contents.",
                "tool_calls": [call],
            },
        )
        validate_conversation(messages)

    def test_reasoning_content_is_assistant_only_and_must_be_a_string(self) -> None:
        call = self.tool_call("call-1", "read_file", "{}")
        for message in (
            {"role": "assistant", "content": "x", "reasoning_content": 1},
            {"role": "assistant", "content": "x", "reasoning_content": None},
            {"role": "user", "content": "x", "reasoning_content": "no"},
            {"role": "system", "content": "x", "reasoning_content": "no"},
            {"role": "tool", "tool_call_id": "call-1", "content": "x", "reasoning_content": "no"},
        ):
            with self.subTest(role=message["role"]):
                with self.assertRaises(ProjectionError):
                    validate_conversation([message])

        validate_conversation(
            [
                {"role": "assistant", "content": None, "reasoning_content": "think", "tool_calls": [call]},
                {"role": "tool", "tool_call_id": "call-1", "content": "done"},
            ]
        )

    def test_projects_text_path_as_exact_chat_completion_messages(self) -> None:
        self.start_turn("Explain the failure")
        self.apply(
            "assistant_accepted",
            {
                "content": "The parser rejects the malformed record.",
                "finish_reason": "stop",
                "tool_calls": [],
            },
        )

        messages = self.project()

        self.assertEqual(
            messages,
            [
                self.system_message,
                {"role": "user", "content": "Explain the failure"},
                {
                    "role": "assistant",
                    "content": "The parser rejects the malformed record.",
                },
            ],
        )
        self.assertNotIn("/stale/rollout/cwd", messages[0]["content"])

    def test_projects_multi_tool_path_with_provider_ids_and_ignores_runtime_events(
        self,
    ) -> None:
        self.start_turn("Inspect both files")
        first = self.tool_call(
            "provider-1", "read_file", '{"path":"a.py"}'
        )
        second = self.tool_call(
            "provider-2", "grep", '{"pattern":"TODO"}'
        )
        self.apply(
            "assistant_accepted",
            {
                "content": "I will inspect both.",
                "finish_reason": "tool_calls",
                "tool_calls": [first, second],
            },
        )
        first_key = f"3:{first['id']}"
        self.apply(
            "approval_decided",
            {"call_key": first_key, "call_id": "provider-1", "approved": True},
        )
        self.apply(
            "tool_started",
            {"call_key": first_key, "call_id": "provider-1"},
        )
        self.apply(
            "tool_finished",
            {
                "call_key": first_key,
                "call_id": "provider-1",
                "status": "succeeded",
                "result": "a.py: contents",
            },
        )
        self.apply(
            "tool_finished",
            {
                "call_key": "3:provider-2",
                "call_id": "provider-2",
                "status": "invalid_arguments",
                "result": "pattern must be non-empty",
            },
        )
        self.apply(
            "assistant_accepted",
            {"content": "Inspection finished.", "tool_calls": []},
        )

        messages = self.project()

        self.assertEqual(
            messages,
            [
                self.system_message,
                {"role": "user", "content": "Inspect both files"},
                {
                    "role": "assistant",
                    "content": "I will inspect both.",
                    "tool_calls": [first, second],
                },
                {
                    "role": "tool",
                    "tool_call_id": "provider-1",
                    "content": "a.py: contents",
                },
                {
                    "role": "tool",
                    "tool_call_id": "provider-2",
                    "content": "pattern must be non-empty",
                },
                {"role": "assistant", "content": "Inspection finished."},
            ],
        )
        serialized = json.dumps(messages)
        self.assertNotIn("3:provider-1", serialized)
        self.assertNotIn("approval_decided", serialized)
        self.assertNotIn("tool_started", serialized)

    def test_valid_succeeded_result_is_not_recovery_blocked(self) -> None:
        self.start_turn()
        self.apply(
            "assistant_accepted",
            {"tool_calls": [self.tool_call("call-1", "bash", "{}")]},
        )
        self.apply("tool_started", {"call_id": "call-1"})
        self.apply(
            "tool_finished",
            {
                "call_id": "call-1",
                "status": "succeeded",
                "result": "command completed",
            },
        )

        messages = self.project()

        self.assertFalse(self.state.recovery_blocked)
        self.assertEqual(
            messages[-1],
            {
                "role": "tool",
                "tool_call_id": "call-1",
                "content": "command completed",
            },
        )

    def test_provider_call_id_may_be_reused_after_its_result(self) -> None:
        self.start_turn()
        for name, status in (
            ("read_file", "invalid_arguments"),
            ("grep", "not_executed"),
        ):
            self.apply(
                "assistant_accepted",
                {
                    "tool_calls": [self.tool_call("same-id", name, "{}")]
                },
            )
            assistant_seq = self.events[-1].seq
            self.apply(
                "tool_finished",
                {
                    "call_key": f"{assistant_seq}:same-id",
                    "call_id": "same-id",
                    "status": status,
                    "result": status,
                },
            )

        messages = self.project()

        self.assertEqual(
            [
                message.get("tool_call_id")
                for message in messages
                if message["role"] == "tool"
            ],
            ["same-id", "same-id"],
        )

    def test_project_does_not_mutate_state_or_events(self) -> None:
        self.start_turn()
        self.apply("assistant_accepted", {"content": "done", "tool_calls": []})
        expected_state = SessionReducer.replay(tuple(self.events))
        expected_events = tuple(self.events)

        self.project()

        self.assertEqual(self.state, expected_state)
        self.assertEqual(tuple(self.events), expected_events)

    def test_requested_or_started_call_cannot_be_projected(self) -> None:
        for start_call in (False, True):
            with self.subTest(start_call=start_call):
                self.setUp()
                self.start_turn()
                self.apply(
                    "assistant_accepted",
                    {"tool_calls": [self.tool_call("call-1", "bash", "{}")]},
                )
                if start_call:
                    self.apply(
                        "tool_started",
                        {"call_key": "3:call-1", "call_id": "call-1"},
                    )

                with self.assertRaisesRegex(ProjectionError, "unresolved"):
                    self.project()

    def test_unknown_outcome_blocks_projection(self) -> None:
        self.start_turn()
        self.apply(
            "assistant_accepted",
            {"tool_calls": [self.tool_call("call-1", "bash", "{}")]},
        )
        self.apply(
            "tool_started",
            {"call_key": "3:call-1", "call_id": "call-1"},
        )
        self.apply(
            "tool_finished",
            {
                "call_key": "3:call-1",
                "call_id": "call-1",
                "status": "outcome_unknown",
                "result": "execution outcome is unknown",
                "recovery_blocked": True,
            },
        )

        with self.assertRaisesRegex(ProjectionBlockedError, "recovery"):
            self.project()

    def test_pending_recovery_intent_blocks_until_matching_turn_finish(
        self,
    ) -> None:
        self.start_turn()
        self.apply(
            "assistant_accepted",
            {"tool_calls": [self.tool_call("call-1", "bash", "{}")]},
        )
        self.apply(
            "turn_recovery_intent",
            {
                "turn_id": self.turn_id,
                "action": "recover_interrupted",
                "reason": "resume found an unstarted call",
            },
        )
        self.apply(
            "tool_finished",
            {
                "call_key": "3:call-1",
                "call_id": "call-1",
                "status": "not_executed",
                "result": "tool call was not started before recovery",
                "recovery_blocked": False,
            },
        )

        with self.assertRaisesRegex(
            ProjectionBlockedError, "pending recovery intent"
        ):
            self.project()

        self.apply(
            "turn_finished",
            {
                "turn_id": self.turn_id,
                "status": "interrupted",
                "error": "resume found an unstarted call",
            },
        )
        messages = self.project()
        validate_conversation(messages)
        self.assertEqual(
            [message for message in messages if message["role"] == "tool"],
            [
                {
                    "role": "tool",
                    "tool_call_id": "call-1",
                    "content": "tool call was not started before recovery",
                }
            ],
        )

    def test_reconciled_unknown_projects_one_explicit_recovery_result(self) -> None:
        self.start_turn()
        self.apply(
            "assistant_accepted",
            {"tool_calls": [self.tool_call("call-1", "bash", "{}")]},
        )
        self.apply(
            "tool_started",
            {"call_key": "3:call-1", "call_id": "call-1"},
        )
        self.apply(
            "tool_finished",
            {
                "call_key": "3:call-1",
                "call_id": "call-1",
                "status": "outcome_unknown",
                "result": "execution outcome is unknown",
                "recovery_blocked": True,
            },
        )
        self.apply(
            "tool_reconciled",
            {
                "call_key": "3:call-1",
                "call_id": "call-1",
                "outcome": "succeeded",
                "note": "verified the output file",
            },
        )

        messages = self.project()

        self.assertEqual(
            messages[-1],
            {
                "role": "tool",
                "tool_call_id": "call-1",
                "content": (
                    "User confirmed after recovery that the tool succeeded. "
                    "Note: verified the output file"
                ),
            },
        )
        self.assertEqual(
            sum(message.get("role") == "tool" for message in messages), 1
        )

    def test_call_id_only_recovery_projects_one_reconciled_result(self) -> None:
        self.start_turn()
        self.apply(
            "assistant_accepted",
            {"tool_calls": [self.tool_call("c", "bash", "{}")]},
        )
        self.apply("tool_started", {"call_id": "c"})
        self.apply(
            "tool_finished",
            {
                "call_id": "c",
                "status": "outcome_unknown",
                "result": "execution outcome is unknown",
                "recovery_blocked": True,
            },
        )
        self.apply(
            "tool_reconciled",
            {
                "call_id": "c",
                "outcome": "succeeded",
                "note": "verified after restart",
            },
        )

        replayed = SessionReducer.replay(self.events)
        messages = PromptProjector.project(
            self.events, replayed, self.environment
        )

        self.assertEqual(
            [message for message in messages if message["role"] == "tool"],
            [
                {
                    "role": "tool",
                    "tool_call_id": "c",
                    "content": (
                        "User confirmed after recovery that the tool succeeded. "
                        "Note: verified after restart"
                    ),
                }
            ],
        )

    def test_invalid_event_facts_raise_projection_error(self) -> None:
        self.start_turn()

        with self.assertRaises(ProjectionError):
            PromptProjector.project(
                reversed(self.events), self.state, self.environment
            )

    def test_environment_rejects_extra_fields_instead_of_leaking_them(self) -> None:
        self.start_turn()

        with self.assertRaises(ProjectionError):
            PromptProjector.project(
                self.events,
                self.state,
                {
                    "cwd": "/live/worktree",
                    "platform": "linux",
                    "date": "2026-08-27",
                    "is_git": True,
                    "api_key": "must-not-be-projected",
                },
            )


class CheckpointProjectionTests(ProjectionTestCase):
    def test_checkpoint_baseline_and_suffix_are_validated_as_one_conversation(
        self,
    ) -> None:
        self.start_turn("original task")
        call = self.tool_call("call-1", "read_file", '{"path":"a.py"}')
        self.apply("assistant_accepted", {"tool_calls": [call]})
        self.apply(
            "tool_started",
            {"call_key": "3:call-1", "call_id": "call-1"},
        )
        self.apply(
            "tool_finished",
            {
                "call_key": "3:call-1",
                "call_id": "call-1",
                "status": "succeeded",
                "result": "file contents",
            },
        )
        self.apply(
            "compaction_checkpoint",
            {
                "through_seq": 5,
                "summary": "The original task is still active.",
                "replacement_conversation": [
                    {"role": "user", "content": "compressed task"},
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [call],
                    },
                    {
                        "role": "tool",
                        "tool_call_id": "call-1",
                        "content": "file contents",
                    },
                ],
            },
        )
        expected_system = dict(self.system_message)
        expected_system["content"] += (
            "\nCompacted conversation summary:\n"
            "The original task is still active."
        )

        messages = self.project()

        self.assertEqual(
            messages,
            [
                expected_system,
                {"role": "user", "content": "compressed task"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [call],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call-1",
                    "content": "file contents",
                },
            ],
        )
        self.assertEqual(
            json.dumps(messages, ensure_ascii=False).count(
                "The original task is still active."
            ),
            1,
        )

    def test_latest_checkpoint_replaces_older_baseline_and_appends_only_suffix(
        self,
    ) -> None:
        self.start_turn("old task")
        self.apply("assistant_accepted", {"content": "old answer", "tool_calls": []})
        self.apply(
            "compaction_checkpoint",
            {
                "through_seq": 3,
                "summary": "old summary",
                "replacement_conversation": [
                    {"role": "user", "content": "old baseline"},
                    {"role": "assistant", "content": "old answer"},
                ],
            },
        )
        self.apply(
            "compaction_checkpoint",
            {
                "through_seq": 3,
                "summary": "latest summary",
                "replacement_conversation": [
                    {"role": "user", "content": "latest baseline"},
                    {"role": "assistant", "content": "latest answer"},
                ],
            },
        )
        self.apply(
            "turn_finished",
            {"turn_id": self.turn_id, "status": "completed"},
        )
        next_turn = str(uuid.uuid4())
        self.apply(
            "turn_started",
            {"turn_id": next_turn, "user_input": "suffix task"},
        )
        self.apply(
            "assistant_accepted",
            {"content": "suffix answer", "tool_calls": []},
        )
        expected_system = dict(self.system_message)
        expected_system["content"] += (
            "\nCompacted conversation summary:\nlatest summary"
        )

        messages = self.project()

        self.assertEqual(
            messages,
            [
                expected_system,
                {"role": "user", "content": "latest baseline"},
                {"role": "assistant", "content": "latest answer"},
                {"role": "user", "content": "suffix task"},
                {"role": "assistant", "content": "suffix answer"},
            ],
        )
        serialized = json.dumps(messages)
        self.assertNotIn("old summary", serialized)
        self.assertNotIn("old baseline", serialized)
        self.assertNotIn("compaction_checkpoint", serialized)

    def test_invalid_checkpoint_replacement_is_rejected(self) -> None:
        self.start_turn()
        self.apply("assistant_accepted", {"content": "boundary", "tool_calls": []})
        with self.assertRaisesRegex(
            DomainError, "replacement_conversation.*result"
        ):
            self.apply(
                "compaction_checkpoint",
                {
                    "through_seq": 3,
                    "summary": "summary",
                    "replacement_conversation": [
                        {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                self.tool_call(
                                    "orphaned-call", "read_file", "{}"
                                )
                            ],
                        }
                    ],
                },
            )

    def test_checkpoint_rejects_noncanonical_flat_tool_call(self) -> None:
        self.start_turn()
        self.apply("assistant_accepted", {"content": "boundary", "tool_calls": []})
        with self.assertRaisesRegex(
            DomainError, "replacement_conversation.*(canonical|fields)"
        ):
            self.apply(
                "compaction_checkpoint",
                {
                    "through_seq": 3,
                    "summary": "summary",
                    "replacement_conversation": [
                        {"role": "user", "content": "compressed task"},
                        {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "flat-call",
                                    "name": "read_file",
                                    "arguments": "{}",
                                }
                            ],
                        },
                        {
                            "role": "tool",
                            "tool_call_id": "flat-call",
                            "content": "contents",
                        },
                    ],
                },
            )


class ConversationValidationTests(unittest.TestCase):
    def test_projection_reexports_the_low_level_validator_and_error_type(
        self,
    ) -> None:
        self.assertEqual(validate_conversation.__module__, "mca.conversation")

        from mca.conversation import (
            ConversationError,
            validate_conversation as canonical_validator,
        )

        self.assertIs(validate_conversation, canonical_validator)
        self.assertIs(ProjectionError, ConversationError)
        self.assertTrue(issubclass(ProjectionBlockedError, ProjectionError))

    def test_rejects_non_string_roles_with_projection_error(self) -> None:
        for role in ({}, []):
            with self.subTest(role=role):
                with self.assertRaisesRegex(ProjectionError, "invalid role"):
                    validate_conversation([{"role": role, "content": "x"}])

    def test_rejects_more_than_one_system_message(self) -> None:
        messages = [
            {"role": "system", "content": "current environment"},
            {"role": "system", "content": "stale checkpoint context"},
        ]

        with self.assertRaisesRegex(ProjectionError, "system"):
            validate_conversation(messages)

    def test_rejects_duplicate_call_ids_within_one_assistant(self) -> None:
        call = {
            "id": "duplicate",
            "type": "function",
            "function": {"name": "read_file", "arguments": "{}"},
        }
        messages = [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [call, dict(call)],
            }
        ]

        with self.assertRaisesRegex(ProjectionError, "duplicate"):
            validate_conversation(messages)

    def test_rejects_orphan_or_duplicate_tool_results(self) -> None:
        orphan = [
            {"role": "tool", "tool_call_id": "missing", "content": "x"}
        ]
        call = {
            "id": "call-1",
            "type": "function",
            "function": {"name": "read_file", "arguments": "{}"},
        }
        duplicate = [
            {"role": "assistant", "content": None, "tool_calls": [call]},
            {"role": "tool", "tool_call_id": "call-1", "content": "x"},
            {"role": "tool", "tool_call_id": "call-1", "content": "y"},
        ]

        with self.assertRaisesRegex(ProjectionError, "orphan"):
            validate_conversation(orphan)
        with self.assertRaisesRegex(ProjectionError, "orphan|duplicate"):
            validate_conversation(duplicate)

    def test_rejects_new_user_or_assistant_before_all_results(self) -> None:
        call = {
            "id": "call-1",
            "type": "function",
            "function": {"name": "read_file", "arguments": "{}"},
        }
        for next_message in (
            {"role": "user", "content": "continue"},
            {"role": "assistant", "content": "continue"},
        ):
            with self.subTest(role=next_message["role"]):
                with self.assertRaisesRegex(ProjectionError, "before.*result"):
                    validate_conversation(
                        [
                            {
                                "role": "assistant",
                                "content": None,
                                "tool_calls": [call],
                            },
                            next_message,
                        ]
                    )


class RequestBudgetTests(unittest.TestCase):
    def test_tool_schemas_are_counted(self) -> None:
        messages = [{"role": "system", "content": "Be precise."}]
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read a workspace file",
                    "parameters": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                        "required": ["path"],
                    },
                },
            }
        ]

        without_tools = estimate_request_tokens(messages)
        with_tools = estimate_request_tokens(messages, tools)

        self.assertGreater(with_tools, without_tools)
        self.assertEqual(with_tools, estimate_request_tokens(messages, tools))

    def test_non_ascii_text_uses_a_conservative_deterministic_estimate(self) -> None:
        ascii_estimate = estimate_request_tokens(
            [{"role": "user", "content": "abcdefgh"}]
        )
        chinese_estimate = estimate_request_tokens(
            [{"role": "user", "content": "编程智能体测试请求"}]
        )

        self.assertGreaterEqual(chinese_estimate, ascii_estimate)
        self.assertGreaterEqual(chinese_estimate, 8)

    def test_output_reserve_and_safety_margin_reduce_available_input(self) -> None:
        messages = [{"role": "user", "content": "small request"}]
        estimate = estimate_request_tokens(messages)

        self.assertTrue(
            request_fits_budget(
                messages,
                context_window=estimate + 10,
                reserved_output_tokens=0,
                safety_margin=0,
            )
        )
        self.assertFalse(
            request_fits_budget(
                messages,
                context_window=estimate + 10,
                reserved_output_tokens=7,
                safety_margin=4,
            )
        )


if __name__ == "__main__":
    unittest.main()
