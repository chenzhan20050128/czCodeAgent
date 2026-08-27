# mca Coding Agent Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Build a runnable Python coding agent whose local runtime owns tool execution, append-only session facts, context projection, bounded termination, recovery, compaction, approval, and managed-file undo.

**Architecture:** A single-session controller appends every accepted fact to a JSONL RolloutStore, then applies it to an in-memory SessionState. AgentLoop projects that state into Chat Completions messages, accepts only complete streamed responses, and routes tool calls through one ToolExecutor. Rollout remains the session-execution fact source; model messages and recovery state are derived views.

**Tech Stack:** Python 3.11+, standard library, httpx, unittest, OpenAI-compatible Chat Completions API. No agent framework or Agent SDK.

---

## File map

| Path | Responsibility |
|---|---|
| pyproject.toml | package metadata, mca entry point, httpx dependency |
| src/mca/config.py | environment-based model and runtime configuration |
| src/mca/domain.py | events, tool calls, sampling outcomes, session state |
| src/mca/store.py | locked append-only JSONL, fsync, corruption checks |
| src/mca/projection.py | reducer plus provider-message projection and validation |
| src/mca/sse.py | incremental SSE and streamed tool-call assembly |
| src/mca/model.py | one Chat Completions sample and bounded retry policy |
| src/mca/tools/registry.py | explicit ToolSpec, schema validation, result truncation |
| src/mca/tools/filesystem.py | read/list/write/edit preparation and atomic writes |
| src/mca/tools/search.py | bounded ripgrep search; reports a stable error when rg is missing |
| src/mca/tools/shell.py | approved shell execution, output drain, process-group cleanup |
| src/mca/executor.py | validation, approval, snapshot, execution, event pipeline |
| src/mca/undo.py | hash-checked managed-file undo |
| src/mca/compact.py | complete-group tail selection and checkpoint creation |
| src/mca/agent.py | bounded Turn state machine and recovery reconciliation |
| src/mca/cli.py | one-shot CLI, REPL, resume, compact, undo |
| tests/ | deterministic unit, integration, crash-window, and CLI tests |
| demo/buggy_calculator/ | fixed, reproducible real-model demo fixture |

## Task 1: Package skeleton and configuration

**Files:** create pyproject.toml, src/mca/__init__.py, src/mca/__main__.py, src/mca/config.py, tests/test_config.py.

- [ ] Write a failing unittest proving Config.from_env defaults to https://api.deepseek.com and model deepseek-v4-flash, rejects a missing key for live mode, accepts MCA_API_KEY before DEEPSEEK_API_KEY, and never includes the key in repr(config).
- [ ] Run python3 -m unittest tests.test_config -v; expect import failure because mca.config does not exist.
- [ ] Add a frozen Config dataclass with base_url, api_key, model, context_window, max_output_tokens, max_steps, max_tool_calls_per_batch, request_timeout, max_attempts, retry_budget_seconds, verbose, and yolo. Read secrets only from env and use a redacted custom repr.
- [ ] Add a minimal pyproject.toml using a src layout, Python >=3.11, dependency httpx>=0.27,<1, and console script mca=mca.cli:main.
- [ ] Run the config test and python3 -m compileall -q src; expect success.
- [ ] Commit: chore: scaffold mca package and configuration.

## Task 2: Append-only facts and state reduction

**Files:** create src/mca/domain.py, src/mca/store.py, tests/test_store.py, tests/test_reducer.py.

- [ ] Write failing store tests for sequential seq, JSONL round-trip, file mode 0600, ignoring exactly one non-newline invalid tail, rejecting middle corruption or seq gaps, UUID session-id validation, and exclusive writer lock.
- [ ] Run python3 -m unittest tests.test_store -v; expect missing-module failure.
- [ ] Implement immutable Event, ToolCall, SamplingOutcome, ToolStatus, TurnStatus, FileSnapshot, and mutable derived SessionState. Implement reduce_event(state, event) as a pure function.
- [ ] Implement RolloutStore create/open/append/load/close using O_APPEND, one JSON object per line, flush+fsync, monotonic seq, fcntl.flock, 0700 session directory, and strict middle-record validation.
- [ ] Run store tests; expect success.
- [ ] Write failing reducer tests for accepted assistant calls, requested/started/terminal calls, active/completed turns, latest checkpoint, first file snapshot per turn, updated after-hash, and started without terminal becoming outcome_unknown on recovery.
- [ ] Implement reducer transitions and explicit invalid-transition errors; run both test modules and expect success.
- [ ] Commit: feat: add durable session event log and reducer.

## Task 3: Prompt projection and compaction structure

**Files:** create src/mca/projection.py, tests/test_projection.py.

- [ ] Write failing tests proving projection emits current system context, user turns, assistant tool calls and exactly one result per call; blocks on unknown outcomes; rejects orphan or duplicate results; and uses the latest checkpoint plus only events after through_seq.
- [ ] Add fixtures with one text response and one multi-tool response. Assert exact Chat Completions message dictionaries, including standard tool_calls shape.
- [ ] Run python3 -m unittest tests.test_projection -v; expect missing implementation.
- [ ] Implement PromptProjector.project(state, environment) and validate_conversation(messages). Treat rollout as the session fact source while current cwd/date/Git state remain explicit live inputs.
- [ ] Add estimate_request_tokens accounting for system, serialized messages, tool schemas, structural overhead, output reserve, and safety margin; label it heuristic.
- [ ] Run tests and commit: feat: project durable facts into model context.

## Task 4: SSE assembler and model client

**Files:** create src/mca/sse.py, src/mca/model.py, tests/test_sse.py, tests/test_model.py.

- [ ] Write failing SSE tests for UTF-8 split bytes, comments and blank events, multiple data lines, DONE, content deltas, missing first-chunk id or name, and interleaved tool indexes.
- [ ] Implement incremental UTF-8 SSE framing and per-index accumulators. Require a valid finish event plus DONE; preserve raw argument strings until tool validation.
- [ ] Run SSE tests; expect success.
- [ ] Write failing model tests with httpx.MockTransport: complete text, tool batch, stop-with-tools, empty response, length/filter, HTTP 429 then success, partial stream then fresh retry, retry exhaustion, and Authorization never appearing in exceptions.
- [ ] Implement ModelClient.sample(messages, tools, allow_tools) returning a typed SamplingOutcome. Each attempt owns a new accumulator; no assistant event is written here. Retry only before a response is accepted, honor Retry-After when valid, otherwise exponential backoff with jitter and total budget.
- [ ] Run both test modules and commit: feat: parse streamed model responses with bounded retry.

## Task 5: Explicit tools and safe filesystem operations

**Files:** create src/mca/tools/__init__.py, src/mca/tools/registry.py, src/mca/tools/filesystem.py, src/mca/tools/search.py, tests/test_tool_registry.py, tests/test_filesystem_tools.py, tests/test_search_tool.py.

- [ ] Write failing registry tests for required fields, primitive types, bounds, additionalProperties=false, malformed JSON, unknown tools, and head/tail result truncation.
- [ ] Implement explicit ToolSpec declarations used both for model advertisement and runtime validation; do not derive full schema from annotations.
- [ ] Write failing filesystem tests for read pagination and line numbers, bounded list, atomic write, unique edit, zero or multiple edit matches, absolute path rejection, .., common-prefix traps, and symlink escape.
- [ ] Implement a workspace resolver accepting only relative paths, using canonical/commonpath checks, rejecting symlinks for write/edit, and rechecking path plus before-hash before os.replace.
- [ ] Implement write/edit preparation objects with exact diff, optional first snapshot, and execution closure; use same-directory temp file, flush, fsync, mode preservation, and atomic replace.
- [ ] Write and pass grep tests for bounded rg argv mode and a stable missing-ripgrep error.
- [ ] Run all Task 5 tests and commit: feat: add bounded workspace file and search tools.

## Task 6: Approval, shell, executor, and managed undo

**Files:** create src/mca/approval.py, src/mca/tools/shell.py, src/mca/executor.py, src/mca/undo.py, tests/test_approval.py, tests/test_shell.py, tests/test_executor.py, tests/test_undo.py.

- [ ] Write failing approval tests for allow-once, deny, EOF or Ctrl-C fail-closed, and yolo bypassing only interaction.
- [ ] Implement ApprovalRequest rendering for file path and diff, and shell command and cwd; do not add persistent authorization rules.
- [ ] Write failing shell tests for exit code, stdout/stderr drain beyond pipe capacity, timeout, child-process cleanup, stdin=DEVNULL, truncation, and removal of MCA_API_KEY and DEEPSEEK_API_KEY from child env.
- [ ] Implement Popen with /bin/sh -lc, start_new_session=True, and two drain threads; on timeout/interruption signal the process group, wait, escalate, and reap.
- [ ] Write failing executor tests for event order: approval decision, optional snapshot, fsynced tool_started, effect, terminal tool_finished. Cover invalid args, unknown tool, denial, exception, success, and after_hash.
- [ ] Implement ToolExecutor with one terminal result per accepted call and no automatic side-effect retry.
- [ ] Write failing undo tests for original file, created file, multiple edits using first baseline, external hash conflict, resume-derived snapshot, and partial failure reporting.
- [ ] Implement all-or-conflict preflight, managed undo and undo_finished; run Task 6 tests and commit: feat: add approval execution pipeline and managed undo.

## Task 7: Agent Turn state machine

**Files:** create src/mca/agent.py, tests/test_agent.py.

- [ ] Write failing tests with a scripted fake model for text completion; read-result-text; multiple calls in response order; failure or denial still closing the batch; stop-with-tools; empty, filtered and protocol failure; and context overflow routing to compaction once.
- [ ] Write a failing MAX_STEPS test: after the configured tool batches, one final sample receives no tools and tool_choice=none; attempted calls cannot re-enter the loop; terminal status is max_steps_reached.
- [ ] Write a failing batch-limit test: only first 8 calls execute and remaining calls get batch_limit_exceeded results.
- [ ] Implement AgentLoop as the only owner of Turn transitions. Append and fsync assistant_accepted before any tool execution; append terminal turn status on every handled exit.
- [ ] Add Ctrl-C tests for sampling, approval, mid-batch not-executed results, and interrupted shell; implement bounded cleanup.
- [ ] Run python3 -m unittest tests.test_agent -v and full suite; commit: feat: implement bounded agent turn state machine.

## Task 8: Compaction, resume, and reconciliation

**Files:** create src/mca/compact.py, update src/mca/agent.py, create tests/test_compact.py, tests/test_resume.py.

- [ ] Write failing compaction tests defining an atomic group as one assistant plus all tool results, or one plain message. Verify groups never split, first user is retained once, tail overlap deduplicates, summary exposes no tools, and checkpoint replacement passes normal projection validation.
- [ ] Implement fixed-section handoff prompt, deterministic old-tool-output shortening, and checkpoint with through_seq, summary, and full replacement conversation. Switch state only after append succeeds.
- [ ] Write failing resume tests for completed session; interrupted turn with requested but unstarted calls closed as not_executed; started without finished entering blocked state; cwd mismatch or missing workspace refusing silent resume.
- [ ] Implement success/failure/abandon reconciliation as tool_reconciled; block model calls until it is fsynced, project confirmed outcomes as tool results, and end the Turn on abandon.
- [ ] Run compaction and resume tests plus full suite; commit: feat: add compacted context and explicit recovery.

## Task 9: CLI, demo, docs, and real API

**Files:** create src/mca/cli.py, tests/test_cli.py, demo/buggy_calculator files, README.md, .env.example, README.txt.

- [ ] Write failing CLI tests for one-shot prompt, REPL commands, resume UUID validation, yolo warning, nonzero failure and interruption exits, and no secret in output.
- [ ] Implement argparse CLI and REPL commands /help, /compact, /undo, /exit. Store sessions under workspace/.mca/sessions, which is gitignored.
- [ ] Add a deterministic calculator fixture with one failing unit test and a one-line implementation bug; record expected patch and test command.
- [ ] Write README.md with architecture, installation, configuration, safety boundaries, tests, and demo. Write .env.example with names only. Keep README.txt under 1000 Chinese characters and include repository URL, run instructions, and verified features only.
- [ ] Run python3 -m unittest discover -s tests -v, python3 -m compileall -q src tests, and python3 -m pip install -e .; expect success.
- [ ] Export the user-provided secret outside repository files as MCA_API_KEY, select MCA_MODEL=deepseek-v4-flash, run a no-tool smoke test, then run the bug fixture in a copied temporary directory. Never place the key in command text, logs, shell history, or a file.
- [ ] Run the same demo three times with fixed parameters; record pass or failure reasons without credentials. Verify a repository secret scan finds no API key or Authorization header.
- [ ] Commit: docs: add runnable demo and submission guide.

## Final review gate

- [ ] Map every requirement to a test or documented downgrade.
- [ ] Run the complete deterministic suite fresh and report exact counts.
- [ ] Run spec compliance review, then code quality and security review; fix and re-run until both approve.
- [ ] Confirm 技术方案.md remains below 600 lines and update completion claims to evidence.
- [ ] Inspect Git history, tracked files, ignore rules, public remote, and credential scan.
- [ ] Push only verified commits before the deadline; do not rewrite pushed history.
