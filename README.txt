mca —— 从零实现的单机受控编程 Agent

仓库地址：git@github.com:chenzhan20050128/czCodeAgent.git

一、这是什么
mca 是不依赖任何 Agent 框架/SDK、只用 Python 与 httpx 实现的命令行编程 Agent。它基于 OpenAI 兼容的 Chat Completions 协议（默认后端 DeepSeek）。核心思想是：模型只负责提出工具调用，真正的机器控制、流式解析、审批、工具执行和终止都由本地运行时掌握。

二、核心机制（也是我最想讲的设计）
1. rollout 是事实，messages 是投影。每一条被接受的事实都以 append-only JSONL 落盘并 fsync，SessionState 由事实归约得到，模型上下文再由状态投影生成，因此不存在两份互相漂移的历史。
2. 完成不是只看 finish_reason。只要结构上还有 tool_calls 就必须执行并回灌；空响应、截断、断流都不算成功。
3. 副作用边界显式化。工具执行顺序为 assistant 落盘 → tool_started 落盘 → 执行 → tool_finished 落盘；崩溃落在中间会被标为 outcome_unknown，恢复时必须人工确认，绝不静默重试。

三、功能
- Agent 循环与有界终止（MAX_STEPS 后只做一次收口采样）
- 六个工具：read_file、list_dir、grep、write_file、edit_file、bash
- 交互审批、文件原子写、基于快照的 /undo
- 自行解析 SSE 与 tool-call 流式参数、有界重试
- 上下文自动/手动压缩（/compact）；用 provider 真实 token usage 锚定压缩触发，只对新增消息做启发式增量，取 max 保证只会更早压缩
- Plan Mode：先研究后动手。软层注入提示引导先规划，硬层在批准前直接拒绝 write_file/edit_file/bash，用 exit_plan_mode 复用审批网关获批退出
- --resume 会话恢复与崩溃对账；只读会话观测：--list、--show、/status
- 凭据只从环境变量读取，不入库、不落日志

四、安装与运行
python3 -m venv .venv
.venv/bin/pip install -e .
export MCA_API_KEY=你的key   （只从环境变量读取，不写入任何文件）
mca "修复 calculator.py 里失败的测试"   单次任务
mca                                      多轮 REPL
mca --resume <session-id>                恢复会话
mca --plan                               以 plan 模式启动（先研究）
mca --list / mca --show <session-id>     只读列出/回放会话
REPL 命令：/help /status /plan[ off] /compact /undo /exit
直接输入 mca 后，第一条任务可写：workspace: /绝对项目路径 | 任务内容。此时才在目标目录建立会话，文件工具、bash 与 /undo 都被锁在该目录；首轮之后不允许切换路径。
交互 REPL 支持多行任务：Enter 只换行，Ctrl+Enter 提交整个任务；部分终端会把 Ctrl+Enter 编码成普通 Enter，此时用 Ctrl+S 提交。终端使用低饱和蓝灰、靛紫、琥珀、青绿、砖红的语义配色；NO_COLOR、TERM=dumb 或非终端输出自动退化为纯文本。

五、测试与演示
.venv/bin/python -m unittest discover -s tests -v  （确定性测试，使用 fake model/SSE）
demo/buggy_calculator/ 提供可重复的端到端 fixture：初始测试必失败，Agent 修好一行后通过。

六、诚实边界
单机单工作区恢复、不恢复进程；压缩有损但保留原始审计；只支持经测试的协议子集；append-only 只能暴露而非消除未知副作用窗口，不保证 exactly-once。
