# Responsive Code DAG CLI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Subagents are explicitly disabled for this repository task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Render a real, responsive text DAG for Code Mode and restore muted provider thinking in the default CLI.

**Architecture:** Keep durable graph projection unchanged. Add a pure bounded layout pipeline to `terminal.py`: rank and order nodes, render a wide or compact horizontal graph, fall back to connected vertical rails when necessary, then apply semantic ANSI spans. `_Console` continues to own only streaming and redraw state.

**Tech Stack:** Python 3.11+, standard-library dataclasses and collections, Unicode box drawing, ANSI 256-color output, unittest.

---

### Task 1: Lock down default thinking visibility

**Files:** modify `tests/test_cli.py`; modify `src/mca/cli.py`.

- [x] Change the non-verbose console test to require `[thinking]` output and run `python3 -m unittest tests.test_cli.ConsoleFormattingTests.test_reasoning_is_shown_without_verbose_mode -v`; expect failure because output is empty.
- [x] Remove only the `self.verbose` gate from `_Console.reasoning`, retaining empty-delta handling, terminal escaping, muted styling, and stream separation.
- [x] Run the default and verbose reasoning formatting tests; expect all to pass.

### Task 2: Specify connected complex-DAG output

**Files:** modify `tests/test_terminal.py`.

- [x] Add a deterministic graph fixture containing two roots that merge, one independent branch, a fan-out, a long edge, and a failed node whose dependent is blocked.
- [x] Add a wide-render test requiring each node token exactly once inside the diagram, visible connected junctions/arrows, no detached `#N ──▶ #M` edge records, and all status/detail data.
- [x] Add compact and very-narrow tests requiring topology to remain connected while verbose labels move into width-fitted detail rows.
- [x] Add plain/ANSI equivalence and per-node color tests, then run `python3 -m unittest tests.test_terminal.CodeGraphRendererTests -v`; expect the new assertions to fail against the existing list renderer.

### Task 3: Implement deterministic graph layout

**Files:** modify `src/mca/terminal.py`.

- [x] Introduce private immutable layout values for ranked real vertices, route-only dummy vertices, positioned nodes, and styled output spans.
- [x] Compute longest-path ranks, expand long edges across adjacent ranks, and apply bounded left-to-right/right-to-left barycentric ordering with ordinal tie-breakers.
- [x] Render adjacent-rank edges into orthogonal gap lanes on a plain character canvas; merge directional segments into `─│┌┐└┘├┤┬┴┼` junctions and place `▶` only at destination ports.
- [x] Select full horizontal, compact horizontal, or vertical-rail layout from the available display width. Fit escaped labels/details before painting, never after connector geometry is built.
- [x] Apply ANSI per node/detail span after geometry. Keep all connector spans muted and make stripping ANSI reproduce the plain render exactly.
- [x] Run the terminal renderer tests until GREEN, then run `python3 -m unittest tests.test_terminal tests.test_cli -v`.

### Task 4: Verify live redraw and documentation

**Files:** modify `tests/test_cli.py`; modify `README.md`.

- [x] Extend the TTY redraw test with a multi-node diamond snapshot and assert that later snapshots contain the connected diagram below approval output without stale lines.
- [x] Update the Code Mode CLI paragraph and example in `README.md` to describe responsive layered/rail layouts and default muted thinking.
- [x] Run CLI, terminal, graph-projection, and inspection tests; expect all to pass.

### Task 5: Complete, review, and integrate

**Files:** all changed files.

- [x] Run `python3 -m unittest discover -s tests -v`, `python3 -m compileall -q src tests`, `git diff --check`, and a high-confidence credential scan.
- [x] Review behavior against every design bullet, inspect ANSI/plain output manually from deterministic fixtures, and resolve every critical or important issue.
- [ ] Commit the feature branch, switch to `main`, merge without rewriting history, rerun the full verification on the merge result, and push `main`.
- [ ] Confirm local `main`, `origin/main`, and the fetched remote SHA are identical and the worktree is clean.
