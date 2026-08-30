# Parallel Tool Execution Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Execute explicitly concurrency-safe sibling tools with bounded overlap while preserving MCA's single-writer rollout, model-order results, approval behavior, and recovery semantics.

**Architecture:** `ToolSpec` gains a host-only, fail-closed concurrency classifier. A new batch scheduler keeps validation, approvals, event appends, reducer updates, and result commits on the main thread; only approved concurrency-safe handler bodies run in worker threads. Exclusive tools form barriers, and completed results wait in indexed slots until all earlier model-order results can be committed.

**Tech Stack:** Python 3.11+, `concurrent.futures.ThreadPoolExecutor`, existing unittest suite and append-only runtime.

---

### Task 1: Concurrency classification and configuration

**Files:**
- Modify: `src/mca/config.py`
- Modify: `src/mca/tools/registry.py`
- Modify: `src/mca/tools/__init__.py`
- Modify: `tests/test_config.py`
- Modify: `tests/test_tool_registry.py`

- [ ] Add failing tests for a positive `MCA_MAX_PARALLEL_TOOL_CALLS`, fail-closed classification, provider-schema isolation, and the built-in `read_file`/`list_dir` opt-in.
- [ ] Run the focused tests and confirm they fail because the fields and classification API do not exist.
- [ ] Add `max_parallel_tool_calls` with default 4 and an internal `ExecutionMode` classifier. Only exact `True` on a side-effect-free tool may return parallel.
- [ ] Mark only `read_file` and `list_dir` concurrency-safe.
- [ ] Run the focused tests until green.

### Task 2: Staged execution and bounded scheduler

**Files:**
- Create: `src/mca/tool_scheduler.py`
- Modify: `src/mca/executor.py`
- Modify: `src/mca/agent.py`
- Modify: `tests/test_agent.py`
- Modify: `tests/test_executor.py`

- [ ] Add failing tests proving safe siblings overlap, exclusive calls form barriers, the configured cap is respected, and results are appended in model order despite out-of-order completion.
- [ ] Add failing interruption coverage proving no new call starts after interruption and all accepted calls receive one terminal result.
- [ ] Split executor work into main-thread preparation/start, worker-only body dispatch, and main-thread result commit without changing the existing serial `execute()` behavior.
- [ ] Implement a bounded rolling scheduler using `ThreadPoolExecutor`; keep all rollout and reducer mutations on the caller thread.
- [ ] Delegate `AgentLoop._execute_batch()` to the scheduler and retain the existing batch-limit behavior.
- [ ] Run executor and agent tests until green.

### Task 3: Recovery and full verification

**Files:**
- Modify: `tests/test_resume.py` if additional multi-start recovery coverage is needed
- Modify: `README.md`
- Modify: `README.txt` only if the final Chinese submission copy remains within its limit

- [ ] Verify multiple durable `STARTED` calls recover independently as `OUTCOME_UNKNOWN` and still block projection until reconciliation.
- [ ] Document that only `read_file` and `list_dir` overlap; writes, shell, search, approvals, event commits, and result projection remain ordered.
- [ ] Run focused tests, the complete unittest suite, and `compileall`.
- [ ] Inspect the `.repo` worktree diff and confirm no unrelated user changes were overwritten.
