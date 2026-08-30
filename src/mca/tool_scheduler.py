"""Bounded scheduling for one accepted model tool-call batch."""

from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from collections.abc import Callable, Sequence

from .domain import ToolStatus
from .executor import AcceptedToolCall, PreparedParallelCall, ToolExecutor
from .tools.registry import ExecutionMode, ToolResult


class ToolBatchScheduler:
    """Overlap safe bodies while keeping control-plane work model-ordered."""

    def __init__(
        self,
        executor: ToolExecutor,
        *,
        max_parallel: int,
        close_calls: Callable[..., None],
    ) -> None:
        if type(max_parallel) is not int or max_parallel < 1:
            raise ValueError("max_parallel must be a positive integer")
        self.executor = executor
        self.registry = executor.registry
        self.max_parallel = max_parallel
        self.close_calls = close_calls

    def execute(self, calls: Sequence[AcceptedToolCall]) -> bool:
        """Execute calls in barrier groups; return whether the batch interrupted."""

        next_call = 0
        while next_call < len(calls):
            call = calls[next_call]
            if (
                self.registry.execution_mode(call.name, call.raw_arguments)
                is ExecutionMode.EXCLUSIVE
            ):
                result = self.executor.execute(call)
                if result.status == ToolStatus.INTERRUPTED.value:
                    self.close_calls(calls[next_call + 1 :])
                    return True
                next_call += 1
                continue

            group_end = next_call
            while group_end < len(calls):
                candidate = calls[group_end]
                if (
                    self.registry.execution_mode(
                        candidate.name, candidate.raw_arguments
                    )
                    is not ExecutionMode.PARALLEL
                ):
                    break
                group_end += 1
            interrupted = self._run_parallel_group(calls[next_call:group_end])
            if interrupted:
                self.close_calls(calls[group_end:])
                return True
            next_call = group_end
        return False

    def _run_parallel_group(
        self, calls: Sequence[AcceptedToolCall]
    ) -> bool:
        slots: list[tuple[PreparedParallelCall, ToolResult] | None] = [
            None for _ in calls
        ]
        in_flight: dict[Future[ToolResult], tuple[int, PreparedParallelCall]] = {}
        next_to_start = 0
        next_to_commit = 0
        interrupted = False

        def commit_ready() -> bool:
            nonlocal next_to_commit
            observed_interruption = False
            while next_to_commit < len(slots):
                slot = slots[next_to_commit]
                if slot is None:
                    break
                prepared, result = slot
                self.executor.commit_parallel(prepared, result)
                observed_interruption = observed_interruption or (
                    result.status == ToolStatus.INTERRUPTED.value
                )
                next_to_commit += 1
            return observed_interruption

        pool = ThreadPoolExecutor(
            max_workers=min(self.max_parallel, len(calls)),
            thread_name_prefix="mca-tool",
        )
        try:
            while next_to_commit < len(calls):
                while (
                    next_to_start < len(calls)
                    and len(in_flight) < self.max_parallel
                ):
                    call = calls[next_to_start]
                    prepared = self.executor.prepare_parallel(call)
                    future = pool.submit(self.executor.dispatch_parallel, prepared)
                    in_flight[future] = (next_to_start, prepared)
                    next_to_start += 1

                try:
                    done, _ = wait(
                        tuple(in_flight), return_when=FIRST_COMPLETED
                    )
                except KeyboardInterrupt:
                    # Calls in this map already have durable tool_started
                    # facts. Stop admission, wait for the bounded read-only
                    # bodies, commit their real results in model order, and
                    # close calls that never started. Interrupts from prepare
                    # or event appends still propagate to AgentLoop recovery.
                    wait(tuple(in_flight))
                    for future, (index, prepared) in tuple(in_flight.items()):
                        in_flight.pop(future)
                        slots[index] = (prepared, future.result())
                    commit_ready()
                    self.close_calls(calls[next_to_start:])
                    return True
                for future in done:
                    index, prepared = in_flight.pop(future)
                    slots[index] = (prepared, future.result())
                interrupted = commit_ready() or interrupted
                if not interrupted:
                    continue
                while in_flight:
                    done, _ = wait(
                        tuple(in_flight), return_when=FIRST_COMPLETED
                    )
                    for future in done:
                        index, prepared = in_flight.pop(future)
                        slots[index] = (prepared, future.result())
                    interrupted = commit_ready() or interrupted
                self.close_calls(calls[next_to_start:])
                return True
            return interrupted
        finally:
            # Python cannot safely kill worker threads. Every exit waits for
            # submitted safe bodies to reach quiescence before the Turn moves.
            pool.shutdown(wait=True, cancel_futures=True)


__all__ = ["ToolBatchScheduler"]
