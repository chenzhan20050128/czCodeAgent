"""Deterministic tests for the CLI terminal presentation and edit primitives."""

from __future__ import annotations

import os
import sys
import unittest
from io import StringIO
from pathlib import Path
from unittest.mock import patch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mca.terminal import (
    MultiLineBuffer,
    TerminalInputError,
    TerminalTheme,
    is_ctrl_enter_sequence,
    read_multiline_prompt,
)


class TerminalThemeTests(unittest.TestCase):
    def test_disabled_theme_returns_plain_text(self) -> None:
        theme = TerminalTheme(enabled=False)

        self.assertEqual(theme.style("hello", "info"), "hello")
        self.assertEqual(theme.label("session"), "session")

    def test_enabled_theme_uses_low_saturation_256_color_sequences(self) -> None:
        theme = TerminalTheme(enabled=True)
        rendered = theme.style("session ready", "info")

        self.assertTrue(rendered.startswith("\x1b["))
        self.assertIn("\x1b[0m", rendered)
        self.assertNotIn("\x1b[31m", rendered)
        self.assertNotIn("\x1b[32m", rendered)
        self.assertNotIn("\x1b[33m", rendered)

    def test_no_color_environment_disables_auto_theme(self) -> None:
        original = os.environ.get("NO_COLOR")
        try:
            os.environ["NO_COLOR"] = "1"
            self.assertFalse(TerminalTheme.auto(isatty=True).enabled)
        finally:
            if original is None:
                os.environ.pop("NO_COLOR", None)
            else:
                os.environ["NO_COLOR"] = original


class MultiLineBufferTests(unittest.TestCase):
    def test_enter_inserts_newline_and_submit_is_separate(self) -> None:
        buffer = MultiLineBuffer()

        for character in "first":
            buffer.insert(character)
        buffer.newline()
        for character in "second":
            buffer.insert(character)

        self.assertEqual(buffer.value, "first\nsecond")

    def test_backspace_removes_the_last_character_across_a_newline(self) -> None:
        buffer = MultiLineBuffer()
        for character in "a\nb":
            buffer.insert(character)
        buffer.backspace()
        buffer.backspace()

        self.assertEqual(buffer.value, "a")

    def test_recognizes_common_ctrl_enter_terminal_sequences(self) -> None:
        self.assertTrue(is_ctrl_enter_sequence("\x1b[13;5u"))
        self.assertTrue(is_ctrl_enter_sequence("\x1b[27;5;13~"))
        self.assertFalse(is_ctrl_enter_sequence("\r"))
        self.assertFalse(is_ctrl_enter_sequence("\x1b\r"))

    def test_multiline_reader_rejects_non_tty_before_touching_terminal_mode(self) -> None:
        class NonTty:
            def isatty(self):
                return False

        with self.assertRaisesRegex(TerminalInputError, "interactive terminal"):
            read_multiline_prompt(input_stream=NonTty())

    def test_raw_reader_keeps_enter_as_newline_and_ctrl_s_submits(self) -> None:
        class TtyInput:
            def isatty(self):
                return True

            def fileno(self):
                return 7

        rendered = StringIO()
        with patch("mca.terminal.termios.tcgetattr", return_value=["old"]), patch(
            "mca.terminal.termios.tcsetattr"
        ), patch("mca.terminal.tty.setraw"), patch(
            "mca.terminal.os.read",
            side_effect=[b"a", b"\r", b"b", b"\x13"],
        ):
            value = read_multiline_prompt(
                input_stream=TtyInput(), output_stream=rendered
            )

        self.assertEqual(value, "a\nb")
        self.assertIn("...  ", rendered.getvalue())

    def test_raw_reader_submits_on_csi_u_ctrl_enter(self) -> None:
        class TtyInput:
            def isatty(self):
                return True

            def fileno(self):
                return 7

        sequence = [b"x", b"\x1b", b"[", b"1", b"3", b";", b"5", b"u"]
        with patch("mca.terminal.termios.tcgetattr", return_value=["old"]), patch(
            "mca.terminal.termios.tcsetattr"
        ), patch("mca.terminal.tty.setraw"), patch(
            "mca.terminal.os.read", side_effect=sequence
        ):
            value = read_multiline_prompt(
                input_stream=TtyInput(), output_stream=StringIO()
            )

        self.assertEqual(value, "x")


if __name__ == "__main__":
    unittest.main()
