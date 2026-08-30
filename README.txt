mca——从零实现的本地编程智能体

Git仓库：https://github.com/chenzhan20050128/czCodeAgent

运行方法：需要Python 3.11+与ripgrep。执行“python3 -m venv .venv”“.venv/bin/pip install -e .”，通过环境变量MCA_API_KEY提供密钥，然后运行“mca --workspace 项目绝对路径 任务描述”；无参数运行可进入多轮REPL。测试命令为“python3 -m unittest discover -s tests -v”。

核心机制：项目未使用任何Agent框架或服务端代码执行工具，自行实现模型循环、SSE解析、工具协议、上下文压缩、审批、持久化和崩溃恢复。append-only JSONL Rollout是唯一事实源，内存状态与模型上下文均由事件归约/投影得到；已开始但结果未知的副作用不会自动重试，而需人工对账。

特色功能：run_code允许模型用受限Python动态组织DAG，组合read_file、list_dir、grep、write_file、edit_file、bash。await表达依赖，gather/execute表达并行；无依赖的读、写和命令均可并发。节点失败后，下游得到结构化UPSTREAM_FAILED且不执行，无关分支继续。并行写使用整文件FileVersion与规范路径FIFO锁，在原子替换前二次校验；冲突返回FILE_STALE_VERSION，绝不静默覆盖。CLI以彩色动态图显示依赖、当前节点、审批、耗时与失败，也可用“mca --show 会话ID --graph”离线重放。

安全边界：路径限制在工作区；文件修改先展示diff；Shell需审批并有超时、截断和进程组清理。Code Mode不用exec/eval，而在空环境、临时目录的隔离子进程中解释受限AST，并设置源码、AST、步骤、集合、节点、并发、输出、墙钟、CPU及可用时的内存限制。它不是容器级沙箱，获批的Shell仍拥有当前用户权限。凭据只从环境变量读取。
