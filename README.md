# mca — a minimal, controllable coding agent

`mca` is a single-machine coding agent built from scratch on the OpenAI-compatible
Chat Completions protocol (default backend: DeepSeek). The model only proposes
tool calls; a local runtime owns streaming, approval, tool execution, and
termination. Every accepted fact is appended to a JSONL log, and the model's
context is a **projection** of that log — never a second, drifting history.

No agent framework or SDK is used. The only runtime dependency is `httpx`.

## Architecture

One idea drives the whole design: **rollout is the fact, messages are a projection.**

```
user input ──> AgentLoop ──> ModelClient (SSE, bounded retry)
                  │
                  ├─ append fact ──> RolloutStore (JSONL, fsync, flock)
                  │                      │
                  │                      └─ SessionReducer ──> SessionState
                  │                                               │
                  └─ ToolExecutor (validate → approve → snapshot →│ execute)
                                                                  │
                       PromptProjector(state, live env) ──────────┘
                                   │
                             Chat messages for the next sample
```

- **domain.py** — immutable `Event`, `SessionState`, and the pure `SessionReducer`.
- **store.py** — single-writer append-only JSONL with `fsync`, an exclusive
  `flock`, and tail-repair on the last torn line.
- **projection.py / conversation.py** — project facts into a strict Chat
  Completions subset and validate the tool-call/result protocol.
- **sse.py / model.py** — incremental SSE parsing and one bounded-retry sample.
- **agent.py** — the bounded Turn state machine (the only owner of turn transitions).
- **tools/** — six explicit tools: `read_file`, `list_dir`, `grep`,
  `write_file`, `edit_file`, `bash`, plus `exit_plan_mode` for plan review.
- **executor.py / approval.py / undo.py** — approval, atomic writes, managed undo.
- **compact.py** — checkpoint generation at a complete sampling boundary.
- **session.py** — resume, crash recovery, and reconciliation of uncertain outcomes.
- **inspect.py** — read-only summaries and transcript rendering over the fact log.
- **cli.py** — the CLI/REPL glue that assembles the above.

## Install

```bash
python3 -m venv .venv
.venv/bin/pip install -e .
```

Requires Python 3.11+.

## Configure

Set the API key in your environment (see `.env.example`); it is read only from
`MCA_API_KEY` (preferred) or `DEEPSEEK_API_KEY`. It is never written to files,
the rollout, command text, or logs.

```bash
export MCA_API_KEY=...          # do this in your shell, not in a file
export MCA_MODEL=deepseek-v4-flash
```

For DeepSeek V4 Flash, mca defaults to its 1M-token context window, a 512,000
token local sample budget, and `thinking=enabled`. DeepSeek's documented API
ceiling is 384,000 output tokens, so mca safely sends `max_tokens=384000` to
that provider and reserves that actual amount in its context calculation.
Override deliberately:

```bash
export MCA_CONTEXT_WINDOW=1000000
export MCA_MAX_OUTPUT_TOKENS=512000
export MCA_THINKING=enabled
export MCA_MAX_STEPS=64
export MCA_REQUEST_TIMEOUT=600
export MCA_RETRY_BUDGET_SECONDS=900
```

## Use

```bash
mca "fix the failing test in calculator.py"   # one-shot task
mca                                            # interactive REPL
mca --resume <session-id>                      # resume a session
mca --list                                     # list sessions (read-only)
mca --show <session-id>                        # print a session transcript (read-only)
mca --plan                                     # start in plan mode (research first)
mca --verbose                                  # stream output + show turn status
mca --yolo                                     # skip approval (checks still apply)
```

When you start bare `mca` outside the target project, its first task binds the
session safely to an explicit absolute workspace instead of treating the launch
directory as the target:

```text
workspace: /absolute/path/to/project | investigate the failing integration tests and fix the root cause
```

This creates the rollout under that project and confines file tools, `bash`,
and `/undo` to the same directory. Binding is allowed only before the first
turn; use `mca --workspace /path/to/project` when you already know the target.

Each approval is `[y/N/always]`. `always` writes an auditable **session-scoped**
authorization: later side-effecting calls in that session auto-approve, and a
resumed session restores that scope. It does not authorize a new session. Run
`/approval reset` to return to per-call prompts.

In an interactive terminal, the REPL is a multi-line editor: **Enter** inserts
a line break and **Ctrl+Enter** submits the whole prompt. mca recognizes the
standard CSI-u / modifyOtherKeys Ctrl+Enter sequences. Some terminals encode
Ctrl+Enter exactly like ordinary Enter; in that case use **Ctrl+S** to submit.
The prompt prints this fallback behavior in `/help`. Colors use a muted
blue/indigo/amber/teal/brick semantic palette and automatically turn off when
stdout is not a TTY, `TERM=dumb`, or `NO_COLOR` is set.
With `--verbose`, provider-supplied reasoning deltas appear as multiline muted
`[thinking]` text; complete tool calls appear in deep amber before
approval/execution, so an active agent does not look stalled. Without
`--verbose`, reasoning stays hidden. mca never invents or exposes reasoning it
did not receive from the provider.

For DeepSeek's long thinking turns, the default request timeout is 600 seconds
and the retry budget is 900 seconds. Do not treat a quiet terminal during a
thinking request as a failed request; use `/status` after the turn returns, or
inspect the session with `mca --show <session-id> --workspace <project>`.

REPL commands: `/help`, `/status`, `/plan [off]`, `/compact`, `/undo`, `/exit`.

Sessions are stored under `<workspace>/.mca/sessions/` (gitignored).
`--list`, `--show`, and `/status` are pure read-only projections of the same
fact log the agent runs on; they need no API key, take no lock, and never
modify the rollout — a live session can be inspected while it is still running.

**Plan mode** is a two-layer guardrail: the system prompt asks the model to
research and propose a plan first (soft layer), and the runtime refuses
`write_file`, `edit_file`, and `bash` until you approve the plan via
`exit_plan_mode` (hard layer). Read-only tools stay available. The state is a
log-only `plan_mode_set` fact, so it survives `--resume`.

**Context budget** is anchored on the provider's real token usage when
available: the last successful response's `total_tokens` prices the history it
covered, and only messages appended after it are estimated heuristically. The
result never dips below the pure heuristic, so anchoring can only trigger
compaction earlier, never skip it.

## Safety boundaries

- The API key stays in environment variables only; shell subprocesses have the
  known model-key variables stripped, and error strings redact known secret values.
- File tools accept only relative paths inside the workspace, reject `..` and
  symlink escapes, and re-check the path and content hash before an atomic replace.
- `bash` runs in its own process group with a timeout; the whole group is
  cleaned up on timeout or Ctrl-C. This is not an OS sandbox — commands still
  run with your user's permissions.
- `/undo` reverts only managed `write_file`/`edit_file` changes and refuses to
  overwrite files changed outside the agent. It cannot undo bash side effects.
- If a crash lands between "tool started" and "tool finished", recovery marks
  the call `outcome_unknown` and blocks until you reconcile it — no silent retry.

## Tests

```bash
.venv/bin/python -m unittest discover -s tests -v
```

The suite is deterministic (fake model and fake SSE); it covers the reducer,
projection, SSE assembly, retry policy, the six tools, approval, shell cleanup,
undo, compaction, resume, and crash-window recovery.

## Demo

See `demo/buggy_calculator/` for a repeatable end-to-end fixture whose test
fails until the agent fixes one line.

## Honest boundaries

- Single machine, single workspace resume; processes are not restored.
- Compaction is model-generated and lossy; the original rollout is always kept.
- Only a tested subset of OpenAI-compatible Chat Completions is supported, not
  every gateway.
- Append-only exposes the uncertain side-effect window; it does not remove it or
  guarantee exactly-once execution.
