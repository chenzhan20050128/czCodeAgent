# Python Code Mode Dynamic DAG Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a constrained Python `run_code` tool whose durable dynamic DAG can concurrently execute every ordinary MCA tool, detect file-version conflicts, propagate dependency failures, and render live in the CLI.

**Architecture:** A subprocess hosts a bounded AST interpreter and submits lazy ToolNode graph closures over JSONL RPC. The parent owns ordered approval, a single event/reducer lane, dynamic DAG scheduling, per-path mutation CAS, recovery, and UI projection. Nested calls reuse the existing tool pipeline and remain log-visible but model-hidden; only the outer curated result reaches Chat Completions.

**Tech Stack:** Python 3.11+, standard library `ast`, `subprocess`, `resource`, `threading`, `concurrent.futures`, JSONL, unittest, existing MCA event and tool runtime.

---

### Task 1: Atomic file-version CAS for concurrent mutations

**Files:** create `src/mca/file_versions.py`; modify `src/mca/tools/filesystem.py`, `src/mca/domain.py`, `src/mca/executor.py`, `src/mca/undo.py`; test `tests/test_filesystem_tools.py`, `tests/test_executor.py`, `tests/test_undo.py`.

- [ ] Write failing tests proving two different-file writes overlap, two same-version same-file writes yield one success and one `FILE_STALE_VERSION`, write-vs-edit has the same result, absent-file creation is one-winner, and shared parent creation is race-safe.
- [ ] Implement immutable `FileVersion` plus a canonical-path FIFO `FileMutationCoordinator`; keep version check, temp publication, and post-write version capture inside the target lock.
- [ ] Add one durable mutation plan per write/edit call and remove snapshot-source coupling while preserving the first Turn baseline and latest successful after-version.
- [ ] Move created-directory cleanup after all undo file operations and verify common parents are removed only when empty.
- [ ] Run the three focused modules and the existing filesystem/undo fault-injection cases.

### Task 2: Durable Code DAG domain and recovery

**Files:** create `src/mca/code_graph.py`; modify `src/mca/domain.py`, `src/mca/conversation.py`, `src/mca/projection.py`, `src/mca/session.py`, `src/mca/inspect.py`; test `tests/test_code_graph.py`, `tests/test_reducer.py`, `tests/test_projection.py`, `tests/test_resume.py`.

- [ ] Write failing tests for planned-node validation, parent ownership, dependency references, cycle rejection, every terminal status, upstream root-failure propagation, and model-hidden nested events.
- [ ] Implement immutable node values and reducer state derived only from `code_run_started`, `code_node_planned`, existing nested lifecycle facts, `code_node_finished`, and `code_run_finished`.
- [ ] Extend recovery so planned nodes close deterministically, started nodes become unknown, nested effects reconcile individually, and the outer run closes failed without replay.
- [ ] Verify cold replay reconstructs the same graph and provider conversation remains valid.

### Task 3: General staged tool pipeline

**Files:** modify `src/mca/executor.py`, `src/mca/tool_scheduler.py`, `src/mca/tools/registry.py`; test `tests/test_executor.py`, `tests/test_agent.py`, `tests/test_code_graph.py`.

- [ ] Write failing tests for nested read/write/edit/grep/bash calls using the same validation, Plan Mode, approval, snapshot, timeout, result normalization, and session-approval semantics as model calls.
- [ ] Generalize execution identities with origin and parent call key; extract prepare, dispatch, and commit without changing the public serial `execute()` behavior.
- [ ] Permit Code Mode to mark all six ordinary tools runnable in parallel while native mode keeps its conservative `ExecutionMode` classifier.
- [ ] Prove RolloutStore and SessionState are touched only by the coordinator thread and every accepted nested call receives one terminal fact.

### Task 4: Constrained Python AST interpreter and protocol

**Files:** create `src/mca/code_ast.py`, `src/mca/code_protocol.py`, `src/mca/code_worker.py`, `src/mca/code_runtime.py`; test `tests/test_code_ast.py`, `tests/test_code_protocol.py`, `tests/test_code_runtime.py`.

- [ ] Write failing syntax/semantic tests for the supported language and explicit rejection of import, file/process access, reflection, definitions, arbitrary attributes/calls, dunder names, unbounded collections, and non-JSON values.
- [ ] Implement an iterative/budgeted AST interpreter with ToolNode, `after`, `gather`, `execute`, `ToolCallError`, `GraphExecutionError`, captured print, and stable source diagnostics.
- [ ] Implement size-bounded JSONL frames and a standalone `-I -S -u` worker with empty environment/cwd, POSIX limits, wall timeout, process-group cleanup, and bounded stderr/output.
- [ ] Prove hostile frames, hot loops, hanging graph calls, worker exit, Ctrl-C, and invalid completions settle with stable error codes.

### Task 5: Dynamic graph scheduler and run_code integration

**Files:** create `src/mca/code_scheduler.py`, `src/mca/code_mode.py`, `src/mca/code_sdk.py`; modify `src/mca/tools/__init__.py`, `src/mca/cli.py`; test `tests/test_code_scheduler.py`, `tests/test_code_mode.py`, `tests/test_cli.py`.

- [ ] Write failing tests for lazy node creation, await/gather/execute, dependencies, independent-branch continuation, ordered approval, all-six-tool execution, same-file stale conflict, parallel shell, and forced execution summary.
- [ ] Implement dynamic Kahn scheduling with a separate Code Mode cap, coordinator-only commits, per-node RPC responses, and root-failure propagation.
- [ ] Register `run_code`; generate a stable Python SDK from ToolSpec schemas, excluding `run_code` and `exit_plan_mode`; inject Code Mode guidance into the system prompt.
- [ ] Ensure outer success/failure is emitted only after worker, nested tasks, event appends, and process cleanup reach quiescence.

### Task 6: Colored live DAG and replay

**Files:** modify `src/mca/code_graph.py`, `src/mca/terminal.py`, `src/mca/cli.py`, `src/mca/inspect.py`; test `tests/test_code_graph.py`, `tests/test_terminal.py`, `tests/test_cli.py`, `tests/test_inspect.py`.

- [ ] Write failing golden tests for colored TTY graphs, state transitions, dependency connectors, current-node highlighting, conflicts, upstream failures, shell warnings, terminal-width truncation, approval pause/resume, NO_COLOR, and stable non-TTY lines.
- [ ] Implement a pure event-to-graph projection and separate ANSI/plain renderers; add bounded redraw state to `_Console` without persisting presentation state.
- [ ] Add compact graph output to ordinary transcript rendering and an expanded `--graph` show option.
- [ ] Verify a live render and cold replay produce the same final graph content after ANSI removal.

### Task 7: Documentation, full verification, and real-provider E2E

**Files:** modify `README.md`, `README.txt`, `.env.example`; add or update deterministic E2E fixtures under `demo/`; test the complete repository.

- [ ] Document Python Code Mode, parallel writes, file-level CAS, DAG failure propagation, CLI graph, configuration, and honest sandbox/Bash limits; keep submission README under 1000 Chinese characters.
- [ ] Run the complete unittest suite and `python3 -m compileall -q src tests`.
- [ ] Run secret scanning without printing secret values.
- [ ] With credentials inherited from the login shell, run a read/aggregate real-model Code Mode smoke, then a copied bug-fix fixture using parallel writes plus dependent tests; record only redacted outcomes.
- [ ] Repeat the deterministic E2E path and verify session replay reproduces the final DAG.

### Task 8: Review and integration

**Files:** all changed files.

- [ ] Perform spec-compliance review, code-quality/concurrency review, and security review; fix every critical or important issue and rerun focused tests.
- [ ] Run the complete suite fresh, inspect `git diff --check`, repository status, commit history, and credential scan.
- [ ] Commit cohesive remaining changes, push `feat/python-code-mode-dag`, merge it into local `main`, rerun full verification on `main`, and push `main` without rewriting history.

