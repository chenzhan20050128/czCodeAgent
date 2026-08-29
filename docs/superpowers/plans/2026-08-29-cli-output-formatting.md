# CLI Output Formatting Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make MCA terminal output readable and prevent duplicate final answers.

**Architecture:** Keep presentation state inside `_Console`. Track whether accepted assistant content was rendered live so `_Runtime.report` can print only non-streamed fallback text.

**Tech Stack:** Python 3.11+, unittest, existing terminal theme helpers.

---

### Task 1: Specify console rendering behavior

**Files:**
- Modify: `tests/test_cli.py`

- [ ] Add a test showing non-verbose reasoning emits no output.
- [ ] Add a test showing verbose reasoning preserves newlines and separates the following tool block.
- [ ] Add a test showing streamed final content appears exactly once after `report`.
- [ ] Run `python -m unittest tests.test_cli -v` and confirm the new assertions fail against the old implementation.

### Task 2: Implement bounded presentation state

**Files:**
- Modify: `src/mca/cli.py`

- [ ] Gate `reasoning` on `verbose` and escape it with `preserve_newlines=True`.
- [ ] Add a console section-transition helper that emits a single blank line between live output blocks.
- [ ] Track whether assistant content was streamed and let `report` consume that state before deciding whether to print `final_text`.
- [ ] Run `python -m unittest tests.test_cli -v` and confirm all CLI tests pass.

### Task 3: Verify and integrate

**Files:**
- Modify: `src/mca/cli.py`
- Modify: `tests/test_cli.py`

- [ ] Run `python -m unittest discover -s tests -v`.
- [ ] Commit the feature branch.
- [ ] Merge `feature/cli-output-formatting` into `main`.
- [ ] Run the full suite again on `main`.
- [ ] Push `main` to `origin`.
