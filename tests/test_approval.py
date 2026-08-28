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
    ApprovalInterrupted,
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
                "Warning: shell commands may start descendant processes; "
                "MCA does not manage background jobs after command completion.\n"
            ),
        )

    def test_shell_render_escapes_terminal_controls_without_changing_raw_values(self) -> None:
        command = (
            "printf '\x1b[31mred\x1b[0m'\rOVER\nnext\t"
            "\x1b]8;;https://evil.invalid\x07link\x1b]8;;\x07\u202e\u2066"
        )
        cwd = "/tmp/work\nlabel\t\x1b]0;owned\x07\u202d"
        request = ApprovalRequest(
            tool_name="bash",
            target=command,
            kind="shell",
            cwd=cwd,
        )

        rendered = request.render()

        self.assertEqual(request.target, command)
        self.assertEqual(request.cwd, cwd)
        for unsafe in ("\x1b", "\x07", "\r", "\t", "\u202e", "\u2066", "\u202d"):
            self.assertNotIn(unsafe, rendered)
        self.assertIn(r"\x1b[31mred\x1b[0m", rendered)
        self.assertIn(r"\rOVER\nnext\t", rendered)
        self.assertIn(r"\x1b]8;;https://evil.invalid\x07link", rendered)
        self.assertIn(r"\u202e\u2066", rendered)
        self.assertIn(r"Cwd: /tmp/work\nlabel\t\x1b]0;owned\x07\u202d", rendered)

    def test_file_render_preserves_only_structural_newlines(self) -> None:
        target = "/tmp/file\nname\t\x1b\u202e.txt"
        diff = (
            "--- old\n+++ new\n@@ -1 +1 @@\n"
            "-safe\rhidden\n+new\t\x1b]8;;x\x07link\b\x85\u2067\n"
        )
        request = ApprovalRequest(
            tool_name="write_file",
            target=target,
            kind="file",
            diff=diff,
            before_hash="abc",
        )

        rendered = request.render()

        self.assertEqual(request.target, target)
        self.assertEqual(request.diff, diff)
        for unsafe in ("\x1b", "\x07", "\r", "\t", "\b", "\x85", "\u202e", "\u2067"):
            self.assertNotIn(unsafe, rendered)
        self.assertIn(r"Path: /tmp/file\nname\t\x1b\u202e.txt", rendered)
        self.assertIn("Diff:\n--- old\n+++ new\n@@ -1 +1 @@\n", rendered)
        self.assertIn(r"-safe\rhidden", rendered)
        self.assertIn(r"+new\t\x1b]8;;x\x07link\b\x85\u2067", rendered)


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

    def test_eof_fails_closed_as_denial(self) -> None:
        def eof(_: str) -> str:
            raise EOFError

        decision = InteractiveApprover(
            input_fn=eof, output_fn=lambda _: None
        ).decide(self.request())
        self.assertIs(decision, ApprovalDecision.DENY)

    def test_keyboard_interrupt_fails_closed_and_remains_distinguishable(self) -> None:
        def interrupt(_: str) -> str:
            raise KeyboardInterrupt

        with self.assertRaises(ApprovalInterrupted):
            InteractiveApprover(
                input_fn=interrupt, output_fn=lambda _: None
            ).decide(self.request())

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
