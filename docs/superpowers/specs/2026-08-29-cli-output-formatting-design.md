# CLI Output Formatting Design

> Superseded for reasoning visibility by `2026-08-30-responsive-code-dag-cli-design.md`: provider reasoning is now shown by default.

## Goal

Make live MCA output readable without changing the agent loop, persisted events, or non-interactive transcript format.

## Behavior

- Provider reasoning is displayed by default; `--verbose` controls additional turn diagnostics.
- Displayed reasoning preserves line breaks while terminal control characters remain escaped.
- Transitions between reasoning, tool calls, approvals, and assistant output have one blank separator line.
- Accepted assistant text that was already streamed is not printed again by the final turn report.
- Assistant text is still printed by the report when no live content was streamed, including deterministic tests and fallback results.
- Tool call arguments remain bounded and escaped.

## Scope

The change is confined to `_Console` output state and `_Runtime.report` in `src/mca/cli.py`, with regression coverage in `tests/test_cli.py`. No dependency, protocol, rollout, or model-client changes are needed.

## Verification

Tests cover verbose gating, multiline reasoning, section spacing, and single rendering of streamed final text. The full unittest suite must pass before merge and again after merging into `main`.
