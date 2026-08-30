"""Deterministic tests for the CLI terminal presentation and edit primitives."""

from __future__ import annotations

import os
import re
import sys
import unittest
import unicodedata
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
import mca.terminal as terminal
from mca.code_graph import CodeGraphNodeView, CodeGraphView


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


class CodeGraphRendererTests(unittest.TestCase):
    def _view(self) -> CodeGraphView:
        return CodeGraphView(
            run_id="run-1",
            description="Update service and tests with a deliberately long description",
            status="failed",
            nodes=(
                CodeGraphNodeView(
                    node_id="n1", ordinal=1, name="write_file",
                    target="src/service.py", dependency_ordinals=(),
                    dependent_ordinals=(3,), status="succeeded",
                    is_current=False, elapsed_ms=21, result="wrote file",
                    blocked_by_ordinals=(), root_failure_ordinals=(),
                ),
                CodeGraphNodeView(
                    node_id="n2", ordinal=2, name="edit_file",
                    target="src/service.py", dependency_ordinals=(),
                    dependent_ordinals=(3,), status="conflict",
                    is_current=False, elapsed_ms=8,
                    result="FILE_STALE_VERSION expected 7a19 observed b61f",
                    blocked_by_ordinals=(), root_failure_ordinals=(2,),
                ),
                CodeGraphNodeView(
                    node_id="n3", ordinal=3, name="bash",
                    target="python3 -m unittest", dependency_ordinals=(1, 2),
                    dependent_ordinals=(), status="upstream_failed",
                    is_current=False, elapsed_ms=None, result="dependency failed",
                    blocked_by_ordinals=(2,), root_failure_ordinals=(2,),
                ),
                CodeGraphNodeView(
                    node_id="n4", ordinal=4, name="grep",
                    target="TODO in src", dependency_ordinals=(),
                    dependent_ordinals=(), status="started",
                    is_current=True, elapsed_ms=4, result=None,
                    blocked_by_ordinals=(), root_failure_ordinals=(),
                ),
            ),
            summary={"planned": 4, "started": 4, "succeeded": 1,
                     "conflict": 1, "upstream_failed": 1,
                     "root_failures": ["n2"]},
            elapsed_ms=29,
            shell_mutation_warning=True,
        )

    def _complex_view(self) -> CodeGraphView:
        def node(
            ordinal,
            name,
            target,
            dependencies,
            dependents,
            status="succeeded",
            *,
            current=False,
            result=None,
            blocked_by=(),
        ):
            return CodeGraphNodeView(
                node_id=f"n{ordinal}", ordinal=ordinal, name=name,
                target=target, dependency_ordinals=dependencies,
                dependent_ordinals=dependents, status=status,
                is_current=current, elapsed_ms=ordinal * 7, result=result,
                blocked_by_ordinals=blocked_by,
                root_failure_ordinals=blocked_by,
            )

        return CodeGraphView(
            run_id="run-complex",
            description="Build service, tests, and verify the complete change",
            status="failed",
            nodes=(
                node(1, "read_file", "src/service.py", (), (3, 6)),
                node(2, "grep", "TODO in src", (), (3,)),
                node(3, "edit_file", "src/service.py", (1, 2), (4, 5),
                     status="started", current=True),
                node(4, "write_file", "tests/test_service.py", (3,), (6,)),
                node(5, "bash", "python3 -m lint", (3,), (7,),
                     status="failed", result="lint failed"),
                node(6, "bash", "python3 -m unittest", (1, 4), ()),
                node(7, "bash", "python3 -m package", (5,), (),
                     status="upstream_failed", result="dependency failed",
                     blocked_by=(5,)),
                node(8, "list_dir", "docs", (), ()),
            ),
            summary={"planned": 8, "started": 8, "succeeded": 5,
                     "failed": 1, "upstream_failed": 1,
                     "root_failures": ["n5"]},
            elapsed_ms=93,
            shell_mutation_warning=False,
        )

    def test_plain_graph_shows_full_dag_state_failures_and_warning(self) -> None:
        render = getattr(terminal, "render_code_graph_plain", None)
        self.assertIsNotNone(render, "terminal needs a stable plain DAG renderer")

        output = render(self._view(), width=100, expanded=True)

        self.assertIn("run_code: Update service and tests", output)
        self.assertIn("DAG", output)
        self.assertIn("▶", output)
        self.assertTrue(any(character in output for character in "├┤┬┴┼"))
        self.assertNotRegex(output, r"│  #\d+ ──▶ #\d+$")
        self.assertIn("CURRENT", output)
        self.assertIn("CONFLICT", output)
        self.assertIn("FILE_STALE_VERSION", output)
        self.assertIn("UPSTREAM_FAILED", output)
        self.assertIn("blocked by #2", output)
        self.assertIn("parallel bash + file mutation", output)
        self.assertIn("1 succeeded", output)

    def test_wide_graph_draws_complex_dependencies_instead_of_listing_edges(self) -> None:
        output = terminal.render_code_graph_plain(
            self._complex_view(), width=180, expanded=True
        )
        diagram = output.split("│  Details", 1)[0]

        self.assertIn("DAG · left to right", diagram)
        self.assertIn("[✓ #1", diagram)
        self.assertIn("▶ [▶ #3", diagram)
        self.assertNotRegex(output, r"(?m)^│  #\d+ ──▶ #\d+$")
        self.assertGreaterEqual(diagram.count("▶"), 6)
        self.assertTrue(any(character in diagram for character in "├┤┬┴┼"))
        for ordinal in range(1, 9):
            self.assertEqual(
                diagram.count(f"#{ordinal}"), 1,
                f"node #{ordinal} should appear once in the diagram",
            )
        self.assertIn("CURRENT", output)
        self.assertIn("lint failed", output)
        self.assertIn("blocked by #5", output)

    def test_compact_and_vertical_layouts_keep_topology_and_fit_width(self) -> None:
        compact = terminal.render_code_graph_plain(
            self._complex_view(), width=64, expanded=True
        )
        vertical = terminal.render_code_graph_plain(
            self._complex_view(), width=30, expanded=True
        )

        self.assertIn("DAG · compact", compact)
        self.assertIn("│  Details", compact)
        self.assertIn("▶", compact)
        self.assertIn("DAG · vertical", vertical)
        self.assertIn("▶", vertical)
        self.assertTrue(any(character in vertical for character in "│├┤┬┴┼"))
        self.assertIn("after #1,#2", vertical)
        self.assertIn("after #1,#4", vertical)
        for rendered, width in ((compact, 64), (vertical, 30)):
            self.assertTrue(
                all(self._display_width(line) <= width for line in rendered.splitlines())
            )
            for ordinal in range(1, 9):
                self.assertIn(f"#{ordinal}", rendered)

    def test_ansi_graph_colors_individual_nodes_without_changing_topology(self) -> None:
        graph = self._complex_view()
        plain = terminal.render_code_graph_plain(graph, width=180, expanded=True)
        colored = terminal.render_code_graph_ansi(graph, width=180, expanded=True)
        stripped = re.sub(r"\x1b\[[0-9;]*m", "", colored)

        self.assertEqual(stripped, plain)
        theme = TerminalTheme(enabled=True)
        self.assertIn(
            theme.style("[✓ #1 read_file(src/service.py)]", "success"), colored
        )
        self.assertIn(
            theme.style("[▶ #3 edit_file(src/service.py)]", "prompt"), colored
        )
        self.assertIn(
            theme.style("[✗ #5 bash(python3 -m lint)]", "failure"), colored
        )
        self.assertIn(
            theme.style("[⊘ #7 bash(python3 -m package)]", "approval"), colored
        )

    def test_ansi_graph_only_adds_style_and_every_line_respects_width(self) -> None:
        plain_render = getattr(terminal, "render_code_graph_plain", None)
        ansi_render = getattr(terminal, "render_code_graph_ansi", None)
        self.assertIsNotNone(plain_render)
        self.assertIsNotNone(ansi_render)
        plain = plain_render(self._view(), width=54, expanded=True)
        colored = ansi_render(self._view(), width=54, expanded=True)

        self.assertIn("\x1b[", colored)
        stripped = re.sub(r"\x1b\[[0-9;]*m", "", colored)
        self.assertEqual(stripped, plain)
        self.assertTrue(all(len(line) <= 54 for line in plain.splitlines()))
        self.assertIn("UPSTREAM_FAILED", plain)
        self.assertIn("CURRENT", plain)

    def test_plain_graph_escapes_terminal_controls_from_untrusted_fields(self) -> None:
        graph = self._view()
        malicious = CodeGraphView(
            run_id=graph.run_id,
            description="safe\x1b[31m\nforged",
            status=graph.status,
            nodes=(
                CodeGraphNodeView(
                    node_id="n1", ordinal=1, name="bash",
                    target="echo ok\r\nforged", dependency_ordinals=(),
                    dependent_ordinals=(), status="failed", is_current=False,
                    elapsed_ms=1, result="bad\x1b[2J\nforged",
                    blocked_by_ordinals=(), root_failure_ordinals=(1,),
                ),
            ),
            summary={"planned": 1, "failed": 1},
            elapsed_ms=1, shell_mutation_warning=False,
        )

        output = terminal.render_code_graph_plain(malicious, width=100, expanded=True)

        self.assertNotIn("\x1b", output)
        self.assertIn(r"\x1b[31m\nforged", output)
        self.assertIn(r"bad\x1b[2J\nforged", output)

    def test_graph_truncation_uses_terminal_columns_for_wide_characters(self) -> None:
        graph = self._view()
        wide = CodeGraphView(
            run_id=graph.run_id, description="中文" * 30,
            status=graph.status, nodes=graph.nodes, summary=graph.summary,
            elapsed_ms=graph.elapsed_ms, shell_mutation_warning=False,
        )

        output = terminal.render_code_graph_plain(wide, width=40, expanded=False)

        self.assertTrue(
            all(self._display_width(line) <= 40 for line in output.splitlines())
        )

    def test_extreme_node_target_is_bounded_before_canvas_layout(self) -> None:
        graph = self._complex_view()
        oversized = CodeGraphView(
            run_id=graph.run_id, description=graph.description, status=graph.status,
            nodes=(
                CodeGraphNodeView(
                    node_id=graph.nodes[0].node_id, ordinal=1, name="bash",
                    target="中" * 100_000, dependency_ordinals=(),
                    dependent_ordinals=(), status="succeeded",
                    is_current=False, elapsed_ms=1, result=None,
                    blocked_by_ordinals=(), root_failure_ordinals=(),
                ),
            ),
            summary={"planned": 1, "succeeded": 1}, elapsed_ms=1,
            shell_mutation_warning=False,
        )

        output = terminal.render_code_graph_plain(oversized, width=100, expanded=False)

        self.assertLess(len(output), 500)
        self.assertIn("…]", output)
        self.assertTrue(
            all(self._display_width(line) <= 100 for line in output.splitlines())
        )

    def test_narrow_details_wrap_every_exact_dependency(self) -> None:
        roots = tuple(
            CodeGraphNodeView(
                node_id=f"n{ordinal}", ordinal=ordinal, name="read_file",
                target=f"{ordinal}.py", dependency_ordinals=(),
                dependent_ordinals=(13,), status="succeeded",
                is_current=False, elapsed_ms=1, result=None,
                blocked_by_ordinals=(), root_failure_ordinals=(),
            )
            for ordinal in range(1, 13)
        )
        sink = CodeGraphNodeView(
            node_id="n13", ordinal=13, name="bash", target="verify",
            dependency_ordinals=tuple(range(1, 13)), dependent_ordinals=(),
            status="planned", is_current=False, elapsed_ms=None, result=None,
            blocked_by_ordinals=(), root_failure_ordinals=(),
        )
        graph = CodeGraphView(
            run_id="wide-fan-in", description="wide fan in", status="active",
            nodes=(*roots, sink), summary={"planned": 13, "succeeded": 12},
            elapsed_ms=1, shell_mutation_warning=False,
        )

        output = terminal.render_code_graph_plain(graph, width=20, expanded=True)
        detail = output.split("│  #13 PLANNED", 1)[1]

        for ordinal in range(1, 13):
            self.assertRegex(detail, rf"(?<!\d)#{ordinal}(?!\d)")
        self.assertTrue(
            all(self._display_width(line) <= 20 for line in output.splitlines())
        )

    def test_maximum_dense_graph_layout_is_deterministic_and_bounded(self) -> None:
        node_count = 64
        dependencies = {
            ordinal: tuple(
                candidate for candidate in range(max(1, ordinal - 5), ordinal)
                if (candidate + ordinal) % 3
            )
            for ordinal in range(1, node_count + 1)
        }
        dependents = {ordinal: [] for ordinal in range(1, node_count + 1)}
        for ordinal, items in dependencies.items():
            for dependency in items:
                dependents[dependency].append(ordinal)
        nodes = tuple(
            CodeGraphNodeView(
                node_id=f"n{ordinal}", ordinal=ordinal, name="bash",
                target=f"stage-{ordinal}",
                dependency_ordinals=dependencies[ordinal],
                dependent_ordinals=tuple(dependents[ordinal]),
                status="succeeded", is_current=False, elapsed_ms=ordinal,
                result=None, blocked_by_ordinals=(), root_failure_ordinals=(),
            )
            for ordinal in range(1, node_count + 1)
        )
        graph = CodeGraphView(
            run_id="dense", description="dense graph", status="succeeded",
            nodes=nodes, summary={"planned": 64, "succeeded": 64},
            elapsed_ms=64, shell_mutation_warning=False,
        )

        first = terminal.render_code_graph_plain(graph, width=30, expanded=False)
        second = terminal.render_code_graph_plain(graph, width=30, expanded=False)

        self.assertEqual(first, second)
        self.assertTrue(
            all(self._display_width(line) <= 30 for line in first.splitlines())
        )
        for ordinal in range(1, node_count + 1):
            self.assertIn(f"#{ordinal}]", first)

    @staticmethod
    def _display_width(line: str) -> int:
        return sum(
            0 if unicodedata.combining(character)
            else 2 if unicodedata.east_asian_width(character) in {"W", "F"}
            else 1
            for character in line
        )


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

    def test_raw_reader_decodes_a_multibyte_utf8_paste_incrementally(self) -> None:
        class TtyInput:
            def isatty(self):
                return True

            def fileno(self):
                return 7

        # "你" arrives as three separate os.read() bytes, exactly as it does
        # in raw terminal mode. It must not become three replacement glyphs.
        with patch("mca.terminal.termios.tcgetattr", return_value=["old"]), patch(
            "mca.terminal.termios.tcsetattr"
        ), patch("mca.terminal.tty.setraw"), patch(
            "mca.terminal.os.read",
            side_effect=[b"\xe4", b"\xbd", b"\xa0", b"\x13"],
        ):
            value = read_multiline_prompt(
                input_stream=TtyInput(), output_stream=StringIO()
            )

        self.assertEqual(value, "你")


if __name__ == "__main__":
    unittest.main()
