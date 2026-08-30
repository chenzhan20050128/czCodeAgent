# Python Code Mode Dynamic DAG Design

## Goal

Add a `run_code` tool that lets the model express a dynamic workflow in a constrained Python language. The program may compose every ordinary MCA tool (`read_file`, `list_dir`, `grep`, `write_file`, `edit_file`, and `bash`), while `exit_plan_mode` and recursive `run_code` calls remain unavailable. The CLI renders the live dependency graph with color and status, and the same graph is reconstructable from the rollout.

## Design principles

1. The model owns business dependency declarations. Nodes without dependency edges may run concurrently, including writes and shell commands.
2. The runtime owns safety. Every nested call still passes schema validation, Plan Mode enforcement, approval, file snapshots, durable start/result facts, timeouts, and recovery.
3. File conflicts never become silent last-writer-wins inside one MCA process. Every file mutation commits under a canonical-path FIFO lock and compares an immutable whole-file `FileVersion`.
4. A failed node prevents its descendants from running. Every skipped descendant receives an `upstream_failed` terminal fact with the root failure chain; unrelated branches continue.
5. The model sees only the outer `run_code` result, including a runtime-authored execution summary. Nested events remain auditable and drive the CLI graph but never become Chat Completions tool messages.
6. A crashed Python program is never replayed. Durable nested calls are reconciled first; the outer `run_code` then closes as failed and the model receives the reconstructed graph summary.

## Model-facing Python contract

`run_code` accepts `code` and a non-empty `description`. `code` is the body of a constrained async Python program. The prompt contains a generated, stable Python SDK for the six callable tools.

```python
write_service = tools.write_file({
    "path": "src/service.py",
    "content": service_source,
})
write_test = tools.write_file({
    "path": "tests/test_service.py",
    "content": test_source,
})
run_tests = tools.bash(
    {"command": "python3 -m unittest -v", "timeout_seconds": 120},
    after=[write_service, write_test],
)
return await run_tests
```

Tool methods create lazy `ToolNode` values. `await node` executes the node and its dependency closure. `await gather(a, b)` executes the union of both closures and returns results in argument order. `execute(*targets)` is the explicit equivalent. A submitted node is immutable and repeated awaits return its cached outcome. Foreign-node references, dependency cycles, and references to an unknown node are rejected before dispatch.

The interpreter supports literals, assignments, `if`, bounded `for`, comprehensions, subscripts, comparisons, boolean and basic arithmetic expressions, formatted strings, `await`, `return`, and `try/except ToolCallError` or `GraphExecutionError`. It rejects imports, direct file/process/network access, function/class/lambda definitions, reflection, dunder access, and arbitrary method dispatch. Only documented pure helpers and the tool SDK are callable. All arguments, intermediate tool results crossing the worker protocol, and program completion values must be lossless JSON.

## Runtime isolation

The evaluator runs in a fresh Python subprocess started with `-I -S -u`, an empty environment, an empty temporary cwd, a separate process group, bounded source/AST/evaluation/tool-node/output sizes, a wall timeout, and POSIX CPU/address-space limits when available. It is a constrained interpreter over `ast`, not `exec` or `eval`. The parent and worker exchange size-bounded JSONL frames. The worker cannot open files or launch processes; it can only submit graph closures to the parent. This is stronger containment than in-process evaluation but is not presented as a container-grade sandbox.

## Graph and scheduling

Each node has `node_id`, `ordinal`, tool name, arguments, dependencies, dependents, state, timing, result/error, and optional file mutation plan. Planning is durable before execution. The parent runs a dynamic Kahn scheduler with a configurable cap. Ready nodes are prepared and approved in ordinal order, then their bodies may overlap. A single coordinator appends every event and mutates `SessionState`.

Unlike native sibling scheduling, Code Mode accepts the model's explicit graph as the concurrency declaration. All six tools may overlap when ready. This includes `write_file`, `edit_file`, and `bash`. Approval remains ordered and each node receives an independent decision. Shell resource conflicts cannot be inferred generally; a parallel frontier containing shell and mutation nodes receives a visible warning but is not blocked.

Terminal states are `succeeded`, `failed`, `denied`, `invalid_arguments`, `conflict`, `timed_out`, `interrupted`, `not_executed`, `upstream_failed`, `outcome_unknown`, and `abandoned`. Any non-success dependency gives an unstarted descendant `upstream_failed`; the error contains direct blockers and deduplicated root failures. Independent branches continue.

## File mutation CAS

`FileVersion` contains existence, whole-file SHA-256, permission mode, size, device, inode, mtime-ns, and ctime-ns. An absent target has a canonical absent version. Every prepared write/edit records the expected version, proposed hash, and approved diff in a per-call mutation plan.

`FileMutationCoordinator` owns one FIFO lock per canonical target path. The mutation critical section performs path/symlink revalidation, reads the current version, compares it with the expected version, creates and fsyncs the temporary file, rechecks the version, atomically replaces the target, fsyncs the parent, and reports the new version. Different targets can commit concurrently. Same-target mutations are ordered by node ordinal; one prepared from a stale version returns `FILE_STALE_VERSION`.

Whole-file hashing remains intentional. Hunk-level rebasing could apply a patch to content different from the approved diff and therefore would require a newly rendered diff and a second approval. It is outside this version. The in-process path lock does not exclude external writers; the second version check narrows that race, and any remaining cross-process rename race is documented rather than described as an atomic filesystem CAS.

The first mutation of a path in a Turn remains the undo baseline. A separate durable mutation plan exists for every call, so whichever concurrent call succeeds may update the path's latest `after_version`. Undo first restores/deletes all eligible files, then deduplicates and removes created directories deepest-first.

## Durable events and recovery

The rollout adds `code_run_started`, `code_node_planned`, `code_node_finished`, and `code_run_finished`. Existing approval, snapshot, start, result, and reconciliation machinery is generalized to model and code origins. Nested events include the parent `run_code` call key and never project as provider tool messages.

On recovery, planned but unstarted nodes become `not_executed` or `upstream_failed`; started nodes without results become `outcome_unknown`. The session remains blocked until every unknown effect is reconciled. Confirming a nested file mutation as successful records its observed after-version so managed undo remains conditional. The outer orchestration call itself is safe to close as failed after nested reconciliation because its only effects are the recorded subcalls; its Python stack is never resumed or replayed.

## Outer result

The program controls its JSON return and captured log messages, but MCA always appends an `execution_summary` containing planned, started, succeeded, failed, denied, conflicted, timed-out, interrupted, upstream-failed, and unknown counts plus root failure details. A program cannot hide a denied or failed node by catching its exception.

## CLI graph

TTY output uses the existing muted palette and redraws one bounded graph block. The current node, approval state, dependency edges, elapsed time, aggregate counts, conflicts, and shell-concurrency warning remain visible. Rendering pauses before an approval prompt and resumes below it so terminal input is never overwritten.

```text
╭─ run_code: Update service and tests                     FAILED
│  #1 write_file  src/service.py               ✓ 21 ms
│  #2 write_file  tests/test_service.py         ✗ CONFLICT
│     FILE_STALE_VERSION expected 7a19c2… observed b61f80…
│  #1 ─┐
│      ├──▶ #3 bash  python3 -m unittest        ⊘ UPSTREAM
│  #2 ─┘      blocked by #2
╰─ 1 succeeded · 1 failed · 1 skipped · wall 28 ms
```

Non-TTY output emits stable `[code-dag]` lines. `mca --show` reconstructs a compact final graph from the same events; an optional `--graph` prints the expanded view. No UI-only graph history is persisted.

## Configuration

- `MCA_CODE_MAX_SOURCE_BYTES=65536`
- `MCA_CODE_MAX_AST_NODES=10000`
- `MCA_CODE_MAX_EVAL_STEPS=100000`
- `MCA_CODE_MAX_WALL_SECONDS=120`
- `MCA_CODE_MAX_CPU_SECONDS=30`
- `MCA_CODE_MAX_MEMORY_MB=256`
- `MCA_CODE_MAX_OUTPUT_BYTES=65536`
- `MCA_CODE_MAX_TOOL_NODES=64`
- `MCA_CODE_MAX_PARALLEL_NODES=4`
- `MCA_CODE_MAX_COLLECTION_ITEMS=10000`

## Verification requirements

Tests cover AST rejection, JSONL framing, runtime budgets, dynamic planning, cycle rejection, gather ordering, dependency propagation, unrelated-branch continuation, all six nested tools, ordered approvals, Plan Mode denial, session always/yolo behavior, same-file stale conflicts, different-file overlap, shared-directory creation, parallel shell cleanup, every crash boundary, nested reconciliation, undo, colored TTY rendering, plain output, cold replay, full regression, and one real-provider end-to-end task.

