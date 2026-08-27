"""Approval rendering and fail-closed interaction tests."""

from __future__ import annotations

import hashlib
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mca.approval import (
    ApprovalDecision,
    ApprovalRequest,
    InteractiveApprover,
)
from mca.tools.filesystem import FileSystemTools


class ApprovalRequestTests(unittest.TestCase):
    def test_file_request_renders_exact_canonical_target_diff_and_hash(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary).resolve()
            path = workspace / "notes.txt"
            path.write_text("old\n", encoding="utf-8")
            prepared = FileSystemTools(workspace).prepare_write_file(
                {"path": "notes.txt", "content": "new\n"}
            )

            request = ApprovalRequest.for_file("write_file", prepared)

        self.assertEqual(request.tool_name, "write_file")
        self.assertEqual(request.target, str(path))
        self.assertEqual(request.before_hash, hashlib.sha256(b"old\n").hexdigest())
        self.assertEqual(request.diff, prepared.diff)
        self.assertEqual(
            request.render(),
            (
                "Tool: write_file\n"
                f"Path: {path}\n"
                f"Before SHA-256: {request.before_hash}\n"
                "Diff:\n"
                f"{prepared.diff}"
            ),
        )

    def test_new_file_request_renders_absent_before_hash(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary).resolve()
            prepared = FileSystemTools(workspace).prepare_write_file(
                {"path": "new.txt", "content": "new\n"}
            )

            request = ApprovalRequest.for_file("write_file", prepared)

        self.assertIn("Before SHA-256: <absent>\n", request.render())

    def test_shell_request_renders_exact_command_shell_and_cwd(self) -> None:
        workspace = Path("/tmp/work space")
        canonical_workspace = workspace.resolve()
        request = ApprovalRequest.for_shell(
            command="printf 'hello world'", cwd=workspace
        )

        self.assertEqual(request.tool_name, "bash")
        self.assertEqual(request.target, "printf 'hello world'")
        self.assertEqual(
            request.render(),
            (
                "Tool: bash\n"
                "Shell: /bin/sh -lc\n"
                f"Cwd: {canonical_workspace}\n"
                "Command:\n"
                "printf 'hello world'\n"
            ),
        )


class InteractiveApproverTests(unittest.TestCase):
    def request(self) -> ApprovalRequest:
        return ApprovalRequest.for_shell(command="pwd", cwd="/tmp/work")

    def test_explicit_yes_allows_once_and_displays_exact_rendering(self) -> None:
        displayed: list[str] = []
        prompts: list[str] = []

        def answer(prompt: str) -> str:
            prompts.append(prompt)
            return "yes"

        decision = InteractiveApprover(
            input_fn=answer, output_fn=displayed.append
        ).decide(self.request())

        self.assertIs(decision, ApprovalDecision.ALLOW_ONCE)
        self.assertEqual(displayed, [self.request().render()])
        self.assertEqual(prompts, ["Allow once? [y/N] "])

    def test_explicit_no_denies(self) -> None:
        decision = InteractiveApprover(
            input_fn=lambda _: "n", output_fn=lambda _: None
        ).decide(
            self.request()
        )
        self.assertIs(decision, ApprovalDecision.DENY)

    def test_invalid_or_empty_input_fails_closed(self) -> None:
        for response in ("", "always", "maybe", " y please"):
            with self.subTest(response=response):
                decision = InteractiveApprover(
                    input_fn=lambda _, value=response: value,
                    output_fn=lambda _: None,
                ).decide(self.request())
                self.assertIs(decision, ApprovalDecision.DENY)

    def test_eof_and_keyboard_interrupt_fail_closed(self) -> None:
        for error in (EOFError(), KeyboardInterrupt()):
            with self.subTest(error=type(error).__name__):
                def interrupt(_: str, raised: BaseException = error) -> str:
                    raise raised

                decision = InteractiveApprover(
                    input_fn=interrupt, output_fn=lambda _: None
                ).decide(
                    self.request()
                )
                self.assertIs(decision, ApprovalDecision.DENY)

    def test_yolo_bypasses_interaction_but_returns_allow_once(self) -> None:
        input_fn = Mock(side_effect=AssertionError("input must not be called"))
        output_fn = Mock(side_effect=AssertionError("output must not be called"))

        decision = InteractiveApprover(
            yolo=True, input_fn=input_fn, output_fn=output_fn
        ).decide(self.request())

        self.assertIs(decision, ApprovalDecision.ALLOW_ONCE)
        input_fn.assert_not_called()
        output_fn.assert_not_called()

    def test_decisions_do_not_cache_across_requests(self) -> None:
        answers = iter(("y", "n"))
        approver = InteractiveApprover(
            input_fn=lambda _: next(answers), output_fn=lambda _: None
        )

        self.assertIs(approver.decide(self.request()), ApprovalDecision.ALLOW_ONCE)
        self.assertIs(approver.decide(self.request()), ApprovalDecision.DENY)


if __name__ == "__main__":
    unittest.main()
