# Responsive Code DAG CLI Design

## Goal

Make Code Mode's live terminal block show the dependency graph itself instead of a node list followed by `#from ──▶ #to` records. Restore provider-supplied thinking as a muted live stream in the default CLI at the same time.

## Confirmed behavior

- A wide terminal renders a left-to-right layered DAG. A node appears once in the diagram and orthogonal connectors show fan-out, fan-in, independent branches, and dependencies that skip a layer.
- A narrower terminal switches node labels from `status + ordinal + tool + target` to compact `status + ordinal` tokens. Tool, target, state, approval, elapsed time, and failure details then appear in a detail list below the diagram.
- If even compact horizontal ranks cannot fit, the renderer uses a vertical rail layout. It is still a connected graph, not a detached edge list.
- Topological rank is the longest dependency path. Nodes within a rank use deterministic crossing-reduction passes with ordinal tie-breaking. Layout never depends on event timestamps or terminal color.
- Live TTY redraws render the complete current snapshot. Existing nodes keep their rank and deterministic ordering as later nodes are planned, minimizing movement.
- Success, current/running, failure, blocked, and neutral nodes receive semantic colors individually. Connectors remain muted so one failed node does not color an entire line.
- Plain output, `NO_COLOR`, and replay use the same layout without ANSI codes. Non-TTY live event lines remain append-only because in-place graph redraw is unavailable there.
- Every emitted line respects terminal display width, including wide Unicode characters and ANSI escape sequences. Labels truncate before topology; connectors are never truncated into a misleading graph.
- Provider `reasoning_content` is visible by default with the existing `[thinking]` muted style. `--verbose` continues to control turn-status diagnostics, not reasoning visibility.

## Layout pipeline

`code_graph.py` remains presentation-neutral. `terminal.py` derives ranks and presentation vertices from `CodeGraphView`, inserts route-only dummy vertices for long edges, applies deterministic barycentric sweeps, and renders a character canvas. Adjacent-rank edges receive orthogonal lanes in the gap; box-drawing junctions are merged from directional connection bits, so crossings and joins remain structurally correct.

The renderer first budgets space for the frame and summary, then attempts full horizontal labels, compact horizontal labels, and finally the vertical rail layout. Node details and failure messages are width-fitted separately. ANSI styling is applied to semantic spans after plain geometry is complete, so visible width and plain/ANSI topology stay identical.

`cli.py` continues to own only redraw lifecycle. It asks the renderer for a snapshot at the current terminal width, clears the previous block, and writes the new block. Approval pause/resume behavior is unchanged.

## Safety and bounds

Code Mode already limits a graph to 64 tool nodes. Layout work is bounded by that node set and its validated dependency edges. The renderer validates dependency references defensively, escapes every model-controlled label before layout, and has no filesystem or process side effects.

## Verification

Golden and structural tests cover a diamond, simultaneous fan-out and fan-in, a long edge with an inserted route vertex, independent branches, a failed root with blocked descendants, compact-width switching, vertical fallback, Unicode display widths, per-node ANSI roles, plain/ANSI topology equivalence, live redraw, replay, and default thinking visibility. The complete unittest suite, compile check, diff check, and credential scan run before merge and again on `main`.
