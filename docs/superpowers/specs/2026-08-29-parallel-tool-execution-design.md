# Parallel Tool Execution Design

## Goal

Reduce latency when one assistant response requests independent read operations without weakening MCA's append-only event ordering, approval boundary, or crash recovery semantics.

## Safety classification

Each `ToolSpec` may provide a host-only `is_concurrency_safe(arguments)` classifier. Only an exact `True` on a side-effect-free tool opts in. Missing tools, invalid arguments, exceptions, non-boolean results, and every side-effecting tool are exclusive. The classifier is not included in the provider schema.

Initially only `read_file` and `list_dir` opt in. `grep`, `write_file`, `edit_file`, `bash`, and `exit_plan_mode` remain exclusive.

## Scheduling

Calls are scanned in model order. Consecutive parallel calls run through a bounded rolling thread pool; each exclusive call runs alone as an ordering barrier. `MCA_MAX_PARALLEL_TOOL_CALLS` controls the overlap and defaults to 4; setting it to 1 preserves serial execution.

Only validated handler bodies run in worker threads. Argument validation, approval, `tool_started`, `tool_finished`, RolloutStore writes, and SessionReducer updates stay on the caller thread. Settled results wait in indexed slots and are committed in model order.

## Interruption and recovery

An interrupted result stops new admission. Already-started calls are drained and committed in model order; calls that were never started become `not_executed`. A process crash may leave several calls in `STARTED`; existing recovery converts each to `OUTCOME_UNKNOWN` and keeps the session blocked until explicit reconciliation.

## Non-goals

- Parallel file writes, edits, shell commands, searches, approvals, or plan transitions.
- Pairwise path-conflict analysis or resource-lock declarations.
- Force-killing Python worker threads.
- Changing the existing recovery or tool-result projection protocol.
