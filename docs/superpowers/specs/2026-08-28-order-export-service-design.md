# mca 真实工程体验：订单导出任务服务

## 目标

为 mca 准备一个可独立运行的维护型后端故障现场。它不是从零生成项目的练习，也不是单行 bug：用户面对的是一套已有的异步订单导出服务，需要通过阅读代码、启动 HTTP 服务、复现并发问题、分析 SQLite 状态、修改多个模块并跑完整回归来完成修复。

体验目录固定为 `/private/tmp/mca-order-export-service`，与 mca 仓库及其会话目录隔离。运行时数据只落在该目录下的 `runtime/`。

## 约束

- 仅 Python 标准库和 SQLite；不能要求 pip 安装、Docker 或联网服务。
- 服务必须能以 `python3 -m app.server` 启动，并绑定 `127.0.0.1`。
- 初始版本的基础 happy path 可用，但必须有可稳定复现的失败集成测试。
- 故障必须来自真实工程边界：事务、并发、状态机、缓存一致性和文件清理；不能靠随机 sleep 才复现。
- 不能把任务答案写进 prompt、README 或测试断言说明中。
- 服务只供本地体验，不进入 mca 的产品功能与运行时。

## 业务模型

用户通过 `POST /exports` 提交一个订单导出请求，携带日期范围与 `Idempotency-Key`。服务创建 `queued` 导出任务；worker 领取任务，查询种子订单数据，生成 CSV，再把任务标为 `completed`。用户用 `GET /exports/{id}` 查看状态；完成后由 `GET /exports/{id}/download` 下载 CSV。

任务状态为 `queued -> running -> completed`，或 `queued/running -> retry_wait -> queued`，最终失败为 `failed`。服务记录 `attempt_count`、`max_attempts`、`lease_owner`、`lease_expires_at` 与结果文件路径。

## 模块边界

```text
app/
  models.py       状态常量、纯状态转换校验、任务数据结构
  repository.py   SQLite schema、事务、CAS 认领、幂等创建、持久化查询
  service.py      HTTP 用例：创建、查询、下载、取消；缓存协调
  worker.py       worker 循环、任务执行、失败注入、重试和 CSV 生成
  cache.py        带 TTL 的线程安全任务视图缓存
  cleanup.py      只清理终态且超过保留期的数据库记录与文件
  server.py       标准库 ThreadingHTTPServer、JSON 路由和生命周期
tests/
  test_api.py         HTTP 状态码、幂等接口和下载契约
  test_worker.py      状态机、重试、CSV 结果
  test_concurrency.py 同 key 并发创建、双 worker 认领
  test_cleanup.py     清理边界、文件系统一致性
scripts/seed_orders.py
README.md
runtime/              SQLite、CSV、日志；不提交
```

`models.py` 不碰 I/O；`repository.py` 是唯一 SQL 所在位置；`worker.py` 通过 repository 领取/完成任务，不能直接更新数据库；`service.py` 不自己拼 SQL；`cleanup.py` 先依据 repository 取得候选，再处理文件。这样 mca 必须沿真实调用链找问题，而不是只改一个万能文件。

## 初始缺陷（给测试，不在用户 prompt 中逐项泄题）

1. 幂等检查先读后写，且 schema 缺少对应唯一约束；两个并发请求可能各自创建任务。
2. worker 用“查出 queued 再 update”为两步操作，缺少 `status='queued'` 条件更新；两个 worker 都可能拿到同一任务。
3. 首次执行错误被错误地写成 `completed`，或错误重试计数/回队顺序不一致。
4. 状态查询缓存只在创建时写入，worker 的状态变化没有失效缓存。
5. 清理器只看创建时间，未要求终态；会删除运行中任务的结果路径，或删除记录前不检查文件结果。

每个缺陷至少有一条确定性测试，组合起来至少 7 个失败断言；不存在随机性和定时竞态。并发测试使用 `threading.Barrier` 控制两个请求/worker 同时到达临界区。

## 对外接口与运行方式

```text
POST /exports                  创建或复用幂等导出任务
GET  /exports/{id}             查询当前任务状态
GET  /exports/{id}/download    下载已完成的 CSV
POST /admin/workers/run-once   手工驱动一个 worker，便于测试与演示
POST /admin/cleanup            手工触发清理器
GET  /healthz                  健康检查
```

启动命令：

```bash
cd /private/tmp/mca-order-export-service
python3 -m app.server --port 8765 --db runtime/exports.db --data-dir runtime/files
```

无三方依赖。测试统一使用：

```bash
python3 -m unittest discover -s tests -v
```

## mca 体验任务

用户将使用一个生产 issue 风格 prompt。它要求 mca 先读 README/测试、启动或复用服务、复现问题，然后修复并保证：

- 同一 `Idempotency-Key` 的并发 POST 只能创建一条记录；
- 一个任务只被一个 worker 成功认领；
- 失败在最大次数内正确重试，成功或耗尽后到达正确终态；
- 查询 API 不读到过期缓存；
- 清理器只处理终态且已过期的任务与对应文件；
- HTTP API 兼容，完整测试和至少一次 curl E2E 通过。

Prompt 明确禁止“删掉并发、禁用缓存、关闭重试或跳过清理”这类绕过。mca 的 `--plan` 可用于先观察它是否只读调查、输出方案，再由 `exit_plan_mode` 放行写工具；普通模式也必须能直接修复。

## 验收证据

生成服务后必须先验证：服务可启动，healthz 可访问，初始完整测试确实失败且失败对应上述边界。之后给用户：

1. 常驻服务进程的 PID、端口和日志路径；
2. 一套 curl 命令创建、查询、触发 worker、下载 CSV；
3. 可直接复制的 mca prompt；
4. 单独的 reset 命令，以便多次体验回到初始故障状态；
5. 运行一次真实 DeepSeek mca 的 rollout 分析，确认它实际跨越 read/search/write/bash/HTTP/测试链路。

## 不做

- 不做认证、真实订单系统、网络消息队列、Docker、多进程守护或外部对象存储。
- 不把服务设计成 mca 的新功能；它只是压力更高、可演示的 E2E fixture。
- 不为制造 token 消耗故意塞无意义的大文件；复杂度来自真实调用关系和反复验证。
