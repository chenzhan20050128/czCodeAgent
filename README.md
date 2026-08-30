# mca：从零实现的本地编程智能体

仓库：https://github.com/chenzhan20050128/czCodeAgent

mca 不依赖 LangChain、OpenAI Agents SDK 等 Agent 框架，也不使用服务端托管的文件或代码执行能力。它只通过 `httpx` 调用 OpenAI 兼容的 Chat Completions API；对话循环、SSE 解析、工具协议、权限控制、持久化、恢复、并发调度和 Code Mode 均在本地实现。

## 快速开始

要求 Python 3.11+，`grep` 工具依赖 `ripgrep`。

```bash
python3 -m venv .venv
.venv/bin/pip install -e .
export MCA_API_KEY=你的密钥
mca --workspace /absolute/path/to/project "修复失败的测试"
```

也可以运行无参数 `mca` 进入多轮 REPL，再使用下面的显式格式绑定项目：

```text
workspace: /absolute/path/to/project | 修复失败的测试
```

常用命令：

- `mca --plan ...`：先调研并生成计划，获批前禁止写文件和执行 Bash。
- `mca --resume <session-id>`：恢复会话；不确定的副作用必须人工对账。
- `mca --list`：列出当前工作区的会话。
- `mca --show <session-id>`：只读回放，包含紧凑 Code Mode 图。
- `mca --show <session-id> --graph`：展开依赖边和失败详情。
- REPL 中 `/undo`：条件式撤销上一轮由 mca 管理的文件修改。

## 核心架构

### 事件溯源 Agent Loop

Rollout 是唯一事实源。用户输入、模型响应、审批、工具开始/结束、Code Mode 节点和 Turn 终态按顺序写入 append-only JSONL，并 `fsync` 后再归约为 `SessionState`。模型上下文由同一批事件重新投影，因此运行时状态、恢复状态和发给模型的历史不会形成三套相互漂移的副本。

本地状态机决定何时继续：只要响应中存在 `tool_calls` 就执行并回传；空响应、截断、过滤和协议错误不会被误判为成功。达到步骤上限时仅允许一次禁用工具的收口采样。

### Python Code Mode 与动态 DAG

模型可调用 `run_code`，用受限 Python 组合全部普通工具：`read_file`、`list_dir`、`grep`、`write_file`、`edit_file`、`bash`。`run_code` 和 `exit_plan_mode` 不允许在 Code Mode 内递归调用。

```python
service = tools.write_file({
    "path": "src/service.py",
    "content": service_source,
})
test = tools.write_file({
    "path": "tests/test_service.py",
    "content": test_source,
})
verify = tools.bash(
    {"command": "python3 -m unittest -v"},
    after=[service, test],
)
return await verify
```

工具调用先生成惰性 `ToolNode`。`await node` 执行其依赖闭包，`await gather(a, b)` 或 `execute(a, b)` 表示并行分支，`after=[...]` 表示依赖。父进程使用动态 Kahn 调度器执行 ready frontier，并受 `MCA_CODE_MAX_PARALLEL_NODES` 限制。模型声明无依赖的读、写和 Bash 可以并行；所有副作用仍逐节点经过参数校验、Plan Mode、审批、快照、执行和结果持久化。

受限语言支持三引号多行字符串且保持内容原样，也支持列表/字典推导、`gather(*nodes)`、条件、循环和 `try/except`。它仍明确拒绝 import、函数/类/lambda 定义、反射、`**kwargs` 和任意方法调用；要生成包含普通 Python 函数或 import 的源码，应把它们放在传给 `write_file` 的字符串中。

一个节点失败后，其后代不会执行，而是得到结构化 `UPSTREAM_FAILED`、直接 blocker 和根失败链；无关分支继续。程序可以捕获工具错误，但外层 `execution_summary` 由 runtime 强制生成，不能隐藏失败或拒绝。一次失败的 Code Run 仍可能已经完成部分写入或命令，因此模型应读取 DAG/summary，只重试失败或跳过的分支。模型下一轮只看到精简后的外层 `run_code` 结果，内部节点仍完整保存在审计日志中。

### 并行写与文件 CAS

不同路径的 `write_file`/`edit_file` 可真正并行。每次准备修改时记录整个文件的 `FileVersion`：存在性、SHA-256、权限、大小、inode、设备号和纳秒时间戳；提交时在规范路径的 FIFO 锁内重新校验，再通过同目录临时文件、`fsync` 和原子替换发布。

同一文件的并行修改不会静默覆盖：先提交者成功，使用旧版本的节点返回 `FILE_STALE_VERSION`，依赖它的节点被跳过。当前使用整文件版本而不是 hunk 级重放，因为把旧补丁自动套到新内容上会改变用户批准过的 diff。锁只协调当前 mca 进程；对外部进程的竞争通过提交前二次校验缩小，但不宣称提供跨进程原子 CAS。

### 实时 DAG 界面

TTY 中会以彩色有界区域持续重画完整 DAG，显示节点状态、当前执行节点、审批状态、依赖边、耗时、冲突、上游失败和实际发生的 Bash/文件并发警告。审批提示出现前图会暂停，后续状态在提示下方恢复，避免覆盖输入。管道或 CI 环境输出稳定的 `[code-dag]` 行；`NO_COLOR=1` 可关闭颜色。

### 恢复、安全与限制

- 文件路径限制在工作区内，拒绝路径穿越和符号链接逃逸；写入展示 diff 并支持受管撤销。
- Shell 有超时、输出截断和进程组清理；执行前仍需批准（`--yolo` 除外）。
- Code Mode 不使用 `exec`/`eval`，而是在 `python -I -S` 子进程中解释受限 AST；禁止 import、文件/网络/进程 API、反射、定义函数/类和私有属性。
- worker 使用空环境和临时 cwd，并限制源码、AST 节点、解释步骤、集合、输出、协议帧、墙钟时间和 CPU；支持的平台还施加地址空间限制。macOS 拒绝 `RLIMIT_AS` 时会明确降级到其余边界。
- 这不是容器级沙箱。尤其 `bash` 是经过本地审批后由父进程执行，拥有当前用户权限；请在可信工作区运行。
- 崩溃后不会重放 Python 程序。已开始但未落结果的副作用标记为 `outcome_unknown`，必须核对；未开始节点关闭为 `not_executed` 或 `upstream_failed`。

## 配置

复制 `.env.example` 的变量到私有 shell 配置中，不要提交真实凭据。主要 Code Mode 默认值：

| 变量 | 默认值 | 含义 |
| --- | ---: | --- |
| `MCA_CODE_MAX_SOURCE_BYTES` | 65536 | 程序源码上限 |
| `MCA_CODE_MAX_AST_NODES` | 10000 | AST 节点上限 |
| `MCA_CODE_MAX_EVAL_STEPS` | 100000 | 解释步骤上限 |
| `MCA_CODE_MAX_WALL_SECONDS` | 120 | 整次 Code Mode 墙钟时间 |
| `MCA_CODE_MAX_CPU_SECONDS` | 30 | worker CPU soft limit |
| `MCA_CODE_MAX_MEMORY_MB` | 256 | 支持平台上的地址空间上限 |
| `MCA_CODE_MAX_OUTPUT_BYTES` | 65536 | 日志与返回值共享的 UTF-8 预算 |
| `MCA_CODE_MAX_TOOL_NODES` | 64 | 一次程序最多工具节点数 |
| `MCA_CODE_MAX_PARALLEL_NODES` | 4 | DAG 最大并行度 |
| `MCA_CODE_MAX_COLLECTION_ITEMS` | 10000 | 容器和循环物化上限 |

其余模型、上下文和网络参数见 [.env.example](.env.example)。

## 测试与演示

```bash
python3 -m unittest discover -s tests -v
python3 -m compileall -q src tests
```

`demo/buggy_calculator` 是单文件修复示例；`demo/parallel_bugfix` 包含两个相互独立的错误，适合演示一次 `run_code` 并行修改两个文件，再由依赖的测试节点统一验证。
