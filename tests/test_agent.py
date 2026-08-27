"""Bounded AgentLoop turn-state-machine tests."""

from __future__ import annotations

import dataclasses
import sys
import tempfile
import unittest
import uuid
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mca.agent import (
    AgentLoop,
    AgentLoopError,
    RecoveryBlockedError,
    TurnResult,
)
from mca.approval import ApprovalDecision
from mca.config import Config
from mca.domain import (
    SamplingOutcome,
    SessionReducer,
    SessionState,
    ToolStatus,
    TurnStatus,
)
from mca.executor import ToolExecutor
from mca.model import SampledToolCall, SamplingResult
from mca.projection import ProjectionEnvironment, PromptProjector
from mca.store import RolloutStore
from mca.tools.registry import SideEffect, ToolRegistry, ToolResult, ToolSpec


EMPTY_SCHEMA = {
    "type": "object",
    "properties": {},
    "required": [],
    "additionalProperties": False,
}
MAX_STEPS_FALLBACK = "Maximum tool steps reached before completion."


@dataclasses.dataclass(frozen=True)
class StreamedSample:
    result: SamplingResult | BaseException
    model_invalidates: bool = False


class ScriptedModel:
    def __init__(
        self,
        *results: (
            SamplingResult
            | StreamedSample
            | BaseException
            | Callable[[], SamplingResult]
        ),
    ) -> None:
        self.results = list(results)
        self.requests: list[
            tuple[list[Mapping[str, Any]], list[Mapping[str, Any]], bool]
        ] = []
        self.callbacks: list[tuple[object, object]] = []

    def sample(
        self,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]],
        allow_tools: bool,
        *,
        on_content: object = None,
        on_invalidate: object = None,
    ) -> SamplingResult:
        self.requests.append((list(messages), list(tools), allow_tools))
        self.callbacks.append((on_content, on_invalidate))
        if not self.results:
            raise AssertionError("unexpected model sample")
        result = self.results.pop(0)
        if isinstance(result, StreamedSample):
            assert callable(on_content)
            on_content("partial")
            if result.model_invalidates:
                assert callable(on_invalidate)
                on_invalidate()
            result = result.result
        if isinstance(result, BaseException):
            raise result
        if callable(result):
            return result()
        return result


class FixedApprover:
    def __init__(self, decision: ApprovalDecision) -> None:
        self.decision = decision
        self.requests: list[object] = []

    def decide(self, request: object) -> ApprovalDecision:
        self.requests.append(request)
        return self.decision


class InterruptingApprover:
    def decide(self, request: object) -> ApprovalDecision:
        raise KeyboardInterrupt


def call(call_id: str, name: str, arguments: str = "{}") -> SampledToolCall:
    return SampledToolCall(
        index=0,
        id=call_id,
        type="function",
        name=name,
        arguments=arguments,
    )


def tool_batch(
    *calls: SampledToolCall,
    content: str = "",
    reasoning: str = "",
    finish_reason: str = "tool_calls",
) -> SamplingResult:
    return SamplingResult(
        SamplingOutcome.VALID_TOOL_BATCH,
        content=content,
        reasoning_content=reasoning,
        tool_calls=calls,
        finish_reason=finish_reason,
    )


def text(content: str, *, reasoning: str = "") -> SamplingResult:
    return SamplingResult(
        SamplingOutcome.COMPLETE_TEXT,
        content=content,
        reasoning_content=reasoning,
        finish_reason="stop",
    )


def tool_spec(
    name: str, handler: Callable[[dict[str, Any]], object]
) -> ToolSpec:
    return ToolSpec(
        name=name,
        description=f"Run {name}.",
        schema=EMPTY_SCHEMA,
        handler=handler,
        side_effect=SideEffect.NONE,
    )


class AgentLoopTestCase(unittest.TestCase):
    def make_runtime(
        self,
        *results: (
            SamplingResult
            | StreamedSample
            | BaseException
            | Callable[[], SamplingResult]
        ),
        specs: Sequence[ToolSpec] = (),
        config: Config | None = None,
        approver: object | None = None,
        compactor: Callable[[], object] | None = None,
        environment: Callable[[], ProjectionEnvironment] | None = None,
        on_content: Callable[[str], None] | None = None,
        on_invalidate: Callable[[], None] | None = None,
    ) -> SimpleNamespace:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        workspace = root / "work"
        workspace.mkdir()
        session_id = str(uuid.uuid4())
        store = RolloutStore.create(root / "sessions", session_id)
        self.addCleanup(store.close)
        state = SessionState()
        event = store.append(
            "session_created",
            {
                "cwd": str(workspace),
                "model": "test-model",
                "context_window": 4096,
            },
        )
        SessionReducer.apply(state, event)
        registry = ToolRegistry(specs, workspace=workspace)
        executor = ToolExecutor(
            registry=registry,
            store=store,
            state=state,
            approver=approver or FixedApprover(ApprovalDecision.ALLOW_ONCE),
            workspace=workspace,
        )
        model = ScriptedModel(*results)
        environment_calls: list[ProjectionEnvironment] = []

        def default_environment() -> ProjectionEnvironment:
            current = ProjectionEnvironment(
                cwd=str(workspace),
                platform="test",
                date=f"2026-08-{27 + len(environment_calls):02d}",
                is_git=False,
            )
            environment_calls.append(current)
            return current

        loop = AgentLoop(
            config=config
            or Config(
                api_key="test",
                model="test-model",
                context_window=4096,
            ),
            store=store,
            state=state,
            model=model,
            executor=executor,
            environment=environment or default_environment,
            compactor=compactor,
            on_content=on_content,
            on_invalidate=on_invalidate,
        )
        return SimpleNamespace(
            root=root,
            workspace=workspace,
            store=store,
            state=state,
            registry=registry,
            executor=executor,
            model=model,
            loop=loop,
            environment_calls=environment_calls,
        )

    @staticmethod
    def append(runtime: SimpleNamespace, event_type: str, payload: dict[str, Any]) -> None:
        event = runtime.store.append(event_type, payload)
        SessionReducer.apply(runtime.state, event)

    @staticmethod
    def terminal_events(runtime: SimpleNamespace) -> list[object]:
        return [
            event for event in runtime.store.load() if event.type == "turn_finished"
        ]


class AgentLoopTests(AgentLoopTestCase):
    def test_duplicate_provider_call_ids_fail_without_persisting_invalid_assistant(
        self,
    ) -> None:
        runtime = self.make_runtime(
            tool_batch(call("duplicate", "ok"), call("duplicate", "ok")),
            text("recovered"),
            specs=[tool_spec("ok", lambda _: ToolResult(title="ok", output="ok"))],
        )

        failed = runtime.loop.run_turn("invalid provider batch")

        self.assertEqual(failed.status, TurnStatus.FAILED)
        self.assertEqual(failed.error, "agent turn failed")
        self.assertEqual(
            [event.type for event in runtime.store.load()],
            ["session_created", "turn_started", "turn_finished"],
        )
        self.assertEqual(SessionReducer.replay(runtime.store.load()), runtime.state)
        self.assertIsNone(runtime.state.active_turn_id)

        recovered = runtime.loop.run_turn("try again")

        self.assertEqual(recovered.status, TurnStatus.COMPLETED)
        self.assertEqual(recovered.final_text, "recovered")
        self.assertEqual(len(self.terminal_events(runtime)), 2)

    def test_text_completion_closes_the_turn_and_forwards_ui_callbacks(self) -> None:
        displayed: list[str] = []
        invalidated: list[str] = []
        on_content = displayed.append
        on_invalidate = lambda: invalidated.append("invalid")
        runtime = self.make_runtime(
            StreamedSample(text("done", reasoning="checked")),
            on_content=on_content,
            on_invalidate=on_invalidate,
        )

        result = runtime.loop.run_turn("finish it")

        self.assertEqual(
            result,
            TurnResult(
                status=TurnStatus.COMPLETED,
                final_text="done",
                turn_id=result.turn_id,
                tool_steps=0,
                error=None,
            ),
        )
        with self.assertRaises(dataclasses.FrozenInstanceError):
            result.status = TurnStatus.FAILED
        events = runtime.store.load()
        self.assertEqual(
            [event.type for event in events],
            [
                "session_created",
                "turn_started",
                "assistant_accepted",
                "turn_finished",
            ],
        )
        self.assertEqual(events[1].payload["user_input"], "finish it")
        self.assertEqual(events[2].payload["content"], "done")
        self.assertEqual(events[2].payload["reasoning_content"], "checked")
        self.assertEqual(events[2].payload["tool_calls"], ())
        self.assertEqual(events[-1].payload["status"], "completed")
        self.assertIsNone(runtime.state.active_turn_id)
        self.assertEqual(displayed, ["partial"])
        self.assertEqual(invalidated, [])
        self.assertIsNot(runtime.model.callbacks[0][0], on_content)
        self.assertIsNot(runtime.model.callbacks[0][1], on_invalidate)

    def test_environment_exception_safely_fails_turn_and_allows_next_turn(self) -> None:
        calls = 0

        def environment() -> ProjectionEnvironment:
            nonlocal calls
            calls += 1
            if calls == 1:
                raise RuntimeError("sensitive environment detail")
            return ProjectionEnvironment(
                cwd="/safe", platform="test", date="2026-08-27", is_git=False
            )

        runtime = self.make_runtime(text("recovered"), environment=environment)

        failed = runtime.loop.run_turn("first")
        recovered = runtime.loop.run_turn("second")

        self.assertEqual((failed.status, failed.error), (TurnStatus.FAILED, "agent turn failed"))
        self.assertEqual(recovered.status, TurnStatus.COMPLETED)
        self.assertEqual(len(self.terminal_events(runtime)), 2)
        self.assertIsNone(runtime.state.active_turn_id)

    def test_projector_exception_safely_fails_turn_and_allows_next_turn(self) -> None:
        runtime = self.make_runtime(text("recovered"))

        with patch.object(
            PromptProjector,
            "project",
            side_effect=RuntimeError("sensitive projector detail"),
        ):
            failed = runtime.loop.run_turn("first")
        recovered = runtime.loop.run_turn("second")

        self.assertEqual((failed.status, failed.error), (TurnStatus.FAILED, "agent turn failed"))
        self.assertEqual(recovered.status, TurnStatus.COMPLETED)
        self.assertEqual(len(self.terminal_events(runtime)), 2)
        self.assertIsNone(runtime.state.active_turn_id)

    def test_model_exception_safely_fails_turn_and_allows_next_turn(self) -> None:
        runtime = self.make_runtime(
            RuntimeError("sensitive model detail"), text("recovered")
        )

        failed = runtime.loop.run_turn("first")
        recovered = runtime.loop.run_turn("second")

        self.assertEqual((failed.status, failed.error), (TurnStatus.FAILED, "agent turn failed"))
        self.assertEqual(recovered.status, TurnStatus.COMPLETED)
        self.assertEqual(len(self.terminal_events(runtime)), 2)
        self.assertIsNone(runtime.state.active_turn_id)

    def test_system_exit_is_not_caught_or_terminalized(self) -> None:
        runtime = self.make_runtime(SystemExit(7))

        with self.assertRaises(SystemExit) as raised:
            runtime.loop.run_turn("exit")

        self.assertEqual(raised.exception.code, 7)
        self.assertEqual(
            [event.type for event in runtime.store.load()],
            ["session_created", "turn_started"],
        )

    def test_read_result_is_projected_exactly_before_final_text(self) -> None:
        executions: list[str] = []

        def read(_: dict[str, Any]) -> ToolResult:
            executions.append("read")
            return ToolResult(title="read", output="1 | hello")

        first = tool_batch(
            call("read-1", "read"),
            content="I will inspect it.",
            reasoning="Need evidence.",
        )
        runtime = self.make_runtime(
            first, text("The file says hello."), specs=[tool_spec("read", read)]
        )

        result = runtime.loop.run_turn("Inspect the file")

        self.assertEqual(result.status, TurnStatus.COMPLETED)
        self.assertEqual(result.final_text, "The file says hello.")
        self.assertEqual(result.tool_steps, 1)
        self.assertEqual(executions, ["read"])
        self.assertEqual(len(runtime.model.requests), 2)
        schemas = runtime.registry.provider_schemas()
        self.assertEqual(runtime.model.requests[0][1:], (schemas, True))
        self.assertEqual(runtime.model.requests[1][1:], (schemas, True))
        second_messages = runtime.model.requests[1][0]
        self.assertEqual(
            second_messages[1:],
            [
                {"role": "user", "content": "Inspect the file"},
                {
                    "role": "assistant",
                    "content": "I will inspect it.",
                    "reasoning_content": "Need evidence.",
                    "tool_calls": [
                        {
                            "id": "read-1",
                            "type": "function",
                            "function": {"name": "read", "arguments": "{}"},
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "read-1",
                    "content": "1 | hello",
                },
            ],
        )
        self.assertIn("2026-08-28", second_messages[0]["content"])
        self.assertEqual(len(runtime.environment_calls), 2)
        self.assertEqual(
            [event.type for event in runtime.store.load()],
            [
                "session_created",
                "turn_started",
                "assistant_accepted",
                "tool_started",
                "tool_finished",
                "assistant_accepted",
                "turn_finished",
            ],
        )

    def test_multiple_calls_run_in_provider_order_despite_failure_and_denial(
        self,
    ) -> None:
        actions: list[str] = []

        def fail(_: dict[str, Any]) -> ToolResult:
            actions.append("fail")
            raise RuntimeError("boom")

        class PreparedDenied:
            def execute(self) -> ToolResult:
                actions.append("denied-executed")
                return ToolResult(title="denied", output="bad")

        def succeed(_: dict[str, Any]) -> ToolResult:
            actions.append("succeed")
            return ToolResult(title="success", output="ok")

        denied = ToolSpec(
            name="denied",
            description="Needs approval.",
            schema=EMPTY_SCHEMA,
            prepare_handler=lambda _: PreparedDenied(),
            side_effect=True,
            approval_renderer=lambda _: "denied operation",
        )
        runtime = self.make_runtime(
            tool_batch(
                call("one", "fail"),
                call("two", "denied"),
                call("three", "succeed"),
            ),
            text("handled"),
            specs=[tool_spec("fail", fail), denied, tool_spec("succeed", succeed)],
            approver=FixedApprover(ApprovalDecision.DENY),
        )

        result = runtime.loop.run_turn("run all")

        self.assertEqual(result.status, TurnStatus.COMPLETED)
        self.assertEqual(actions, ["fail", "succeed"])
        finished = [
            event for event in runtime.store.load() if event.type == "tool_finished"
        ]
        self.assertEqual(
            [(event.payload["call_id"], event.payload["status"]) for event in finished],
            [("one", "failed"), ("two", "denied"), ("three", "succeeded")],
        )
        self.assertEqual(
            [message["tool_call_id"] for message in runtime.model.requests[1][0][-3:]],
            ["one", "two", "three"],
        )

    def test_stop_finish_reason_with_calls_still_executes_tool_batch(self) -> None:
        executions: list[str] = []
        runtime = self.make_runtime(
            tool_batch(
                call("stop-call", "record"), finish_reason="stop"
            ),
            text("done"),
            specs=[
                tool_spec(
                    "record",
                    lambda _: (
                        executions.append("record"),
                        ToolResult(title="record", output="recorded"),
                    )[1],
                )
            ],
        )

        result = runtime.loop.run_turn("record it")

        self.assertEqual(executions, ["record"])
        self.assertEqual(result.tool_steps, 1)
        self.assertEqual(result.status, TurnStatus.COMPLETED)

    def test_sampling_failures_are_terminal_and_never_retried(self) -> None:
        cases = (
            (
                SamplingOutcome.LENGTH_EXCEEDED,
                "model output exceeded the configured length limit",
                TurnStatus.FAILED,
            ),
            (
                SamplingOutcome.FILTERED,
                "model response was filtered",
                TurnStatus.FAILED,
            ),
            (
                SamplingOutcome.PROTOCOL_ERROR,
                "model returned an invalid response",
                TurnStatus.FAILED,
            ),
            (
                SamplingOutcome.TRANSPORT_INTERRUPTED,
                "model transport was interrupted",
                TurnStatus.FAILED,
            ),
            (
                SamplingOutcome.ABORTED,
                "model sampling was aborted",
                TurnStatus.INTERRUPTED,
            ),
        )
        for outcome, expected_error, status in cases:
            with self.subTest(outcome=outcome):
                runtime = self.make_runtime(
                    SamplingResult(
                        outcome, content="partial", error="provider detail"
                    ),
                    text("must not retry"),
                )

                result = runtime.loop.run_turn("test failure")

                self.assertEqual(result.status, status)
                self.assertEqual(result.error, expected_error)
                self.assertEqual(result.final_text, "")
                self.assertEqual(len(runtime.model.requests), 1)
                self.assertEqual(
                    [event.type for event in runtime.store.load()],
                    ["session_created", "turn_started", "turn_finished"],
                )
                terminal = self.terminal_events(runtime)
                self.assertEqual(len(terminal), 1)
                self.assertEqual(terminal[0].payload["status"], status.value)
                self.assertEqual(terminal[0].payload["error"], expected_error)

    def test_discarded_streamed_length_and_filtered_samples_invalidate_once(self) -> None:
        for outcome in (SamplingOutcome.LENGTH_EXCEEDED, SamplingOutcome.FILTERED):
            with self.subTest(outcome=outcome):
                invalidated: list[str] = []
                runtime = self.make_runtime(
                    StreamedSample(SamplingResult(outcome, content="partial")),
                    on_content=lambda _: None,
                    on_invalidate=lambda: invalidated.append("invalid"),
                )

                result = runtime.loop.run_turn("discard")

                self.assertEqual(result.status, TurnStatus.FAILED)
                self.assertEqual(invalidated, ["invalid"])

    def test_model_invalidation_is_not_duplicated_for_discarded_transport_result(
        self,
    ) -> None:
        invalidated: list[str] = []
        runtime = self.make_runtime(
            StreamedSample(
                SamplingResult(
                    SamplingOutcome.TRANSPORT_INTERRUPTED, content="partial"
                ),
                model_invalidates=True,
            ),
            on_content=lambda _: None,
            on_invalidate=lambda: invalidated.append("invalid"),
        )

        result = runtime.loop.run_turn("transport")

        self.assertEqual(result.status, TurnStatus.FAILED)
        self.assertEqual(invalidated, ["invalid"])

    def test_invalid_streamed_limit_finalization_is_invalidated_once(self) -> None:
        invalidated: list[str] = []
        runtime = self.make_runtime(
            StreamedSample(
                SamplingResult(
                    SamplingOutcome.COMPLETE_TEXT,
                    content="discard me",
                    tool_calls=(call("forbidden", "missing"),),
                    finish_reason="stop",
                )
            ),
            config=Config(api_key="test", max_steps=0),
            on_content=lambda _: None,
            on_invalidate=lambda: invalidated.append("invalid"),
        )

        result = runtime.loop.run_turn("bounded")

        self.assertEqual(result.status, TurnStatus.MAX_STEPS_REACHED)
        self.assertEqual(result.final_text, MAX_STEPS_FALLBACK)
        self.assertEqual(invalidated, ["invalid"])
        self.assertFalse(
            any(event.type == "assistant_accepted" for event in runtime.store.load())
        )

    def test_empty_complete_text_is_a_terminal_protocol_failure(self) -> None:
        runtime = self.make_runtime(text(""), text("must not retry"))

        result = runtime.loop.run_turn("empty")

        self.assertEqual(result.status, TurnStatus.FAILED)
        self.assertEqual(result.error, "model returned an empty text response")
        self.assertEqual(len(runtime.model.requests), 1)
        self.assertEqual(len(self.terminal_events(runtime)), 1)

    def test_context_overflow_compacts_once_then_reprojects_and_succeeds(self) -> None:
        compact_calls: list[str] = []
        runtime = self.make_runtime(
            SamplingResult(SamplingOutcome.CONTEXT_OVERFLOW),
            text("after compaction"),
            compactor=lambda: compact_calls.append("compact") or True,
        )

        result = runtime.loop.run_turn("large context")

        self.assertEqual(result.status, TurnStatus.COMPLETED)
        self.assertEqual(result.final_text, "after compaction")
        self.assertEqual(compact_calls, ["compact"])
        self.assertEqual(len(runtime.model.requests), 2)
        self.assertEqual(len(runtime.environment_calls), 2)

    def test_second_context_overflow_fails_without_second_compaction(self) -> None:
        compact_calls: list[str] = []
        runtime = self.make_runtime(
            SamplingResult(SamplingOutcome.CONTEXT_OVERFLOW),
            SamplingResult(SamplingOutcome.CONTEXT_OVERFLOW),
            text("must not retry"),
            compactor=lambda: compact_calls.append("compact") or True,
        )

        result = runtime.loop.run_turn("large context")

        self.assertEqual(result.status, TurnStatus.FAILED)
        self.assertEqual(
            result.error, "model context overflow persisted after compaction"
        )
        self.assertEqual(compact_calls, ["compact"])
        self.assertEqual(len(runtime.model.requests), 2)
        self.assertEqual(len(self.terminal_events(runtime)), 1)

    def test_failed_compaction_is_a_terminal_failure(self) -> None:
        runtime = self.make_runtime(
            SamplingResult(SamplingOutcome.CONTEXT_OVERFLOW),
            text("must not retry"),
            compactor=lambda: False,
        )

        result = runtime.loop.run_turn("large context")

        self.assertEqual(result.status, TurnStatus.FAILED)
        self.assertEqual(result.error, "context compaction failed")
        self.assertEqual(len(runtime.model.requests), 1)
        self.assertEqual(len(self.terminal_events(runtime)), 1)

    def test_max_steps_gets_exactly_one_no_tools_final_sample_without_reentry(
        self,
    ) -> None:
        executions: list[str] = []

        def record(_: dict[str, Any]) -> ToolResult:
            executions.append("run")
            return ToolResult(title="record", output="ok")

        runtime = self.make_runtime(
            tool_batch(call("one", "record")),
            tool_batch(call("two", "record")),
            tool_batch(
                call("forbidden", "record"), content="I want another tool."
            ),
            text("must not re-enter"),
            specs=[tool_spec("record", record)],
            config=Config(
                api_key="test",
                model="test-model",
                context_window=4096,
                max_steps=2,
            ),
        )

        result = runtime.loop.run_turn("bounded")

        self.assertEqual(result.status, TurnStatus.MAX_STEPS_REACHED)
        self.assertEqual(result.tool_steps, 2)
        self.assertEqual(result.final_text, MAX_STEPS_FALLBACK)
        self.assertEqual(result.error, "maximum tool steps reached")
        self.assertEqual(executions, ["run", "run"])
        self.assertEqual(len(runtime.model.requests), 3)
        self.assertEqual(
            [request[2] for request in runtime.model.requests], [True, True, False]
        )
        self.assertEqual(runtime.model.requests[-1][1], [])
        self.assertFalse(
            any(
                call_state.provider_call_id == "forbidden"
                for call_state in runtime.state.tool_calls.values()
            )
        )
        self.assertEqual(len(self.terminal_events(runtime)), 1)

    def test_max_steps_persists_valid_final_text_but_remains_limit_status(self) -> None:
        runtime = self.make_runtime(
            tool_batch(call("one", "ok")),
            text("Here is the bounded summary."),
            specs=[tool_spec("ok", lambda _: ToolResult(title="ok", output="ok"))],
            config=Config(api_key="test", max_steps=1),
        )

        result = runtime.loop.run_turn("bounded")

        self.assertEqual(result.status, TurnStatus.MAX_STEPS_REACHED)
        self.assertEqual(result.final_text, "Here is the bounded summary.")
        assistants = [
            event for event in runtime.store.load() if event.type == "assistant_accepted"
        ]
        self.assertEqual(len(assistants), 2)
        self.assertEqual(assistants[-1].payload["tool_calls"], ())

    def test_batch_limit_executes_prefix_and_terminalizes_every_extra_once(self) -> None:
        executions: list[str] = []

        def record(arguments: dict[str, Any]) -> ToolResult:
            executions.append(str(arguments["value"]))
            return ToolResult(title="record", output=str(arguments["value"]))

        schema = {
            "type": "object",
            "properties": {"value": {"type": "string"}},
            "required": ["value"],
            "additionalProperties": False,
        }
        spec = ToolSpec("record", "Record value.", schema, handler=record)
        runtime = self.make_runtime(
            tool_batch(
                *(
                    call(str(index), "record", f'{{"value":"{index}"}}')
                    for index in range(4)
                )
            ),
            text("done"),
            specs=[spec],
            config=Config(api_key="test", max_tool_calls_per_batch=2),
        )

        result = runtime.loop.run_turn("limit batch")

        self.assertEqual(result.status, TurnStatus.COMPLETED)
        self.assertEqual(executions, ["0", "1"])
        finished = [
            event for event in runtime.store.load() if event.type == "tool_finished"
        ]
        self.assertEqual(
            [(event.payload["call_id"], event.payload["status"]) for event in finished],
            [
                ("0", "succeeded"),
                ("1", "succeeded"),
                ("2", "batch_limit_exceeded"),
                ("3", "batch_limit_exceeded"),
            ],
        )
        self.assertEqual(len({event.payload["call_key"] for event in finished}), 4)
        self.assertEqual(
            [message["tool_call_id"] for message in runtime.model.requests[1][0][-4:]],
            ["0", "1", "2", "3"],
        )

    def test_keyboard_interrupt_mid_batch_closes_current_and_remaining_calls(self) -> None:
        class Prepared:
            def execute(self) -> ToolResult:
                raise AssertionError("approval interruption must prevent execution")

        spec = ToolSpec(
            "write",
            "Write something.",
            EMPTY_SCHEMA,
            prepare_handler=lambda _: Prepared(),
            side_effect=True,
            approval_renderer=lambda _: "write",
        )
        runtime = self.make_runtime(
            tool_batch(
                call("current", "write"),
                call("remaining-1", "write"),
                call("remaining-2", "write"),
            ),
            text("must not sample"),
            specs=[spec],
            approver=InterruptingApprover(),
        )

        result = runtime.loop.run_turn("interrupt")

        self.assertEqual(result.status, TurnStatus.INTERRUPTED)
        self.assertEqual(result.error, "turn interrupted by user")
        self.assertEqual(len(runtime.model.requests), 1)
        finished = [
            event for event in runtime.store.load() if event.type == "tool_finished"
        ]
        self.assertEqual(
            [(event.payload["call_id"], event.payload["status"]) for event in finished],
            [
                ("current", "cancelled"),
                ("remaining-1", "not_executed"),
                ("remaining-2", "not_executed"),
            ],
        )
        self.assertEqual(len(self.terminal_events(runtime)), 1)

    def test_keyboard_interrupt_during_sampling_finishes_interrupted(self) -> None:
        runtime = self.make_runtime(KeyboardInterrupt(), text("must not sample"))

        result = runtime.loop.run_turn("interrupt sample")

        self.assertEqual(result.status, TurnStatus.INTERRUPTED)
        self.assertEqual(result.error, "turn interrupted by user")
        self.assertEqual(len(runtime.model.requests), 1)
        self.assertEqual(len(self.terminal_events(runtime)), 1)

    def test_keyboard_interrupt_after_terminal_append_preserves_original_finish(
        self,
    ) -> None:
        runtime = self.make_runtime(text("durable answer"))
        real_append = runtime.loop._append
        interrupted = False

        def append_then_interrupt(
            event_type: str, payload: Mapping[str, Any]
        ) -> object:
            nonlocal interrupted
            event = real_append(event_type, payload)
            if event_type == "turn_finished" and not interrupted:
                interrupted = True
                raise KeyboardInterrupt
            return event

        with patch.object(runtime.loop, "_append", side_effect=append_then_interrupt):
            result = runtime.loop.run_turn("finish once")

        self.assertEqual(result.status, TurnStatus.COMPLETED)
        self.assertEqual(result.final_text, "durable answer")
        self.assertIsNone(result.error)
        self.assertEqual(len(self.terminal_events(runtime)), 1)
        self.assertEqual(SessionReducer.replay(runtime.store.load()), runtime.state)
        self.assertIsNone(runtime.state.active_turn_id)

    def test_post_commit_tool_interrupt_keeps_success_and_skips_remaining_calls(
        self,
    ) -> None:
        runtime = self.make_runtime(
            tool_batch(call("committed", "write"), call("remaining", "write")),
            text("must not sample"),
            specs=[tool_spec("write", lambda _: ToolResult(title="write", output="ok"))],
        )

        def execute_then_interrupt(accepted: object) -> object:
            runtime.loop._append(
                "tool_started",
                {
                    "call_key": accepted.call_key,
                    "call_id": accepted.provider_call_id,
                    "name": accepted.name,
                    "arguments": {},
                },
            )
            runtime.loop._append(
                "tool_finished",
                {
                    "call_key": accepted.call_key,
                    "call_id": accepted.provider_call_id,
                    "status": "succeeded",
                    "result": "file commit succeeded",
                    "truncated": False,
                },
            )
            raise KeyboardInterrupt

        with patch.object(runtime.executor, "execute", side_effect=execute_then_interrupt):
            result = runtime.loop.run_turn("write files")

        self.assertEqual(result.status, TurnStatus.INTERRUPTED)
        self.assertEqual(result.error, "turn interrupted by user")
        finished = [
            event for event in runtime.store.load() if event.type == "tool_finished"
        ]
        self.assertEqual(
            [(event.payload["call_id"], event.payload["status"]) for event in finished],
            [("committed", "succeeded"), ("remaining", "not_executed")],
        )
        self.assertEqual(len(self.terminal_events(runtime)), 1)
        self.assertEqual(len(runtime.model.requests), 1)

    def test_recovery_blocked_state_refuses_turn_before_append_or_model_call(self) -> None:
        runtime = self.make_runtime(text("must not sample"))
        old_turn_id = str(uuid.uuid4())
        self.append(
            runtime,
            "turn_started",
            {"turn_id": old_turn_id, "user_input": "old turn"},
        )
        self.append(
            runtime,
            "assistant_accepted",
            {
                "content": None,
                "tool_calls": [
                    {
                        "id": "uncertain",
                        "type": "function",
                        "function": {"name": "missing", "arguments": "{}"},
                    }
                ],
            },
        )
        call_key = f"{runtime.state.last_seq}:uncertain"
        self.append(
            runtime,
            "tool_started",
            {"call_key": call_key, "call_id": "uncertain"},
        )
        self.append(
            runtime,
            "tool_finished",
            {
                "call_key": call_key,
                "call_id": "uncertain",
                "status": "outcome_unknown",
                "result": "outcome requires reconciliation",
                "recovery_blocked": True,
            },
        )
        before = runtime.store.load()

        with self.assertRaisesRegex(RecoveryBlockedError, "recovery is blocked"):
            runtime.loop.run_turn("new turn")

        self.assertEqual(runtime.store.load(), before)
        self.assertEqual(runtime.model.requests, [])

    def test_reducer_failure_after_append_poison_stops_the_loop(self) -> None:
        runtime = self.make_runtime(text("must not sample"))

        with (
            patch("mca.agent.reduce_event", return_value=SessionState()),
            patch(
                "mca.agent.SessionReducer.apply",
                side_effect=RuntimeError("bad reducer"),
            ),
        ):
            with self.assertRaisesRegex(AgentLoopError, "durable event 2"):
                runtime.loop.run_turn("persist first")

        self.assertEqual(
            [event.type for event in runtime.store.load()],
            ["session_created", "turn_started"],
        )
        self.assertEqual(runtime.model.requests, [])
        with self.assertRaisesRegex(AgentLoopError, "unusable"):
            runtime.loop.run_turn("never append again")
        self.assertEqual(len(runtime.store.load()), 2)


if __name__ == "__main__":
    unittest.main()
