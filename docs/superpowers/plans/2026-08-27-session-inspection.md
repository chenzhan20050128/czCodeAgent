# 新特性设计：会话观测（read-only session inspection）

> 状态：设计 → TDD 实现。范围严格限定为“对既有事实日志的只读观测”，不引入新的存储、不改变 agent 循环语义。

## 1. 为什么做这个（讲故事的核心）

本项目最重要的一个设计判断是：

> rollout（append-only JSONL 事实日志）是唯一真相源；模型看到的一切、`--resume` 恢复的一切，都是这份日志经同一个 reducer 得到的**纯投影**。

这个判断最有力的证明，就是：从一个**冷进程、零内存状态**出发，我不仅能重建模型上下文，还能重建任意历史会话的**完整人类可读记录**，并列出每个会话的状态——全部走 agent 运行时用的同一个 `SessionReducer`。没有第二份元数据、没有需要同步的索引。

再叠一个系统层面的故事：因为日志是 append-only，**观测不需要写锁**。写入者只 append，读取者永远看到一个“已提交前缀”；万一读到正在写入的最后半行，直接忽略即可。这正是数据库用 WAL 的同一个理由——append-only 结构同时换来崩溃安全的写和无锁的读。

对答辩：这不是新功能堆料，而是把“事件溯源 + 纯投影”这个中心决策**变成可以现场演示的证据**。

## 2. 做什么（三个面，一个主题）

主题只有一个：对事实日志的只读观测。

- `mca --list`：列出当前工作区下所有会话，一行一个：会话 id、创建时间、模型、Turn 数、最后一个 Turn 状态、工具调用数。
- `mca --show <session-id>`：把某个会话回放成可读记录（用户任务、assistant 文本、每次工具调用及其终态、压缩点、undo、Turn 终态）。
- REPL `/status`：对当前活跃会话打印同样的摘要，并额外给出“当前请求估算 token / 上下文窗口”的预算行，直接呼应压缩机制。

三者共享同一套纯函数，不是三个独立功能。

## 3. 关键设计决策

### 3.1 只读读取路径，不夺写锁

新增 `RolloutStore.read_session_snapshot(sessions_root, session_id) -> list[Event]`：

- 以 `O_RDONLY | O_NOFOLLOW | O_CLOEXEC` 打开，**不获取 flock 写锁**，因此即使目标会话正被另一个进程持有独占写锁，也能读到它已提交的前缀。
- 只读，**绝不** truncate、绝不补写换行、绝不改盘上任何字节。
- 复用与写入者完全相同的逐行校验：中间损坏、seq 跳跃/重复、session 不匹配一律抛 `RolloutCorruptionError`；唯一容忍的是“最后一行是写到一半的残行”，在内存里丢弃即可。

### 3.2 重构而非补丁：抽出纯解析器

把 `_read_and_repair_tail` 里“逐行解析成事件”的核心逻辑抽成纯函数 `_parse_rollout(data, session_id) -> _ParsedRollout`，返回：

- `events`：解析出的合法事件；
- `committed_length`：已提交（带换行的合法事件）字节数——写入者用它决定 truncate 位置；
- `final_needs_newline`：最后一条是合法事件但缺换行——写入者补一个换行；
- `dropped_partial`：最后一条是残行——写入者 truncate，读取者直接丢弃。

写入者（会做盘上修复）和只读读取者（只回内存）都调用它。行为对写入者保持逐字节等价，由既有 store 测试守住不回归。

### 3.3 纯投影渲染，复用 reducer

新增 `src/mca/inspect.py`，全部是纯函数：

- `summarize(state) -> SessionSummary`：从 `SessionState` 派生 id/创建时间/模型/Turn 数/工具数/最后 Turn 状态/是否活跃/是否 recovery-blocked。
- `render_transcript(state) -> str`：按事件顺序渲染成有界的可读文本。
- `list_session_ids(sessions_root) -> list[str]`：列出合法 UUID 命名的 rollout。

CLI 的 `--list/--show` 与 REPL 的 `/status` 都在这之上组装。渲染里的工具参数、结果都截断，避免把一条超长 shell 输出灌进终端。

## 4. 明确不做

- 不做会话删除、改写、导出到外部服务、跨机器同步。
- 不做交互式 TUI 浏览；只做一次性打印。
- `--list/--show` 是纯读，**不需要 API key**，也不触发任何模型请求或副作用。

## 5. 边界与安全

- 只读路径不持有写锁、不改盘，天然对并发的活跃会话安全。
- `session_id` 仍走既有 `_validate_session_id`（拒绝非规范 UUID、路径穿越）。
- 渲染前对每个字段做有界截断；不做通用秘密检测，沿用既有“已知凭据脱敏 + 文件权限”边界。
- 损坏的历史会话：`--show` 报错退出非零，不静默给出半份记录。

## 6. 验收

- 只读读取器：能读活跃（被独占锁）会话；容忍单条残尾；中间损坏/seq 异常/session 不匹配报错；不修改文件字节。
- `summarize/render_transcript`：对空会话、纯文本 Turn、含工具批次、含压缩点、含 undo、含 recovery-blocked 都给出稳定输出。
- CLI：`--list` 无会话时给出明确提示；`--show` 未知/损坏会话报错退出 1；二者无 API key 也能运行。
- REPL `/status`：打印摘要与预算行，不改变会话状态。
- 全量测试在 `PYTHONWARNINGS=error` 下通过。
