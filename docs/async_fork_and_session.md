# Async fork 与 live session 的约束与实现

## 1. Live session 的「严格 user–assistant」是什么

- **协议**：session 是**逐行输入**的。客户端每次往 FIFO 写**一行**，agent 处理完这一轮（可能多步 tool）后打出 `>>> Ready for input.`，此时才能发下一行。
- **不能做的**：**并发**向 session 发多条请求（例如两个进程同时写 FIFO），否则会变成多行被一次读入或顺序错乱，破坏「一行 = 一轮」的假设。
- **结论**：所有自动化测试必须**顺序发送**：`send(cmd1)` → 等 Ready → `send(cmd2)` → 等 Ready，不能并发送多条。

## 2. Async fork 引入的「变量」

- Subagent 在**后台线程**里跑，完成后结果进 `fork_result_queue`。
- **Drain** 会在两种时机发生：  
  (1) **下一轮开始**（用户新发一条消息时）；  
  (2) **等待输入时**（主循环用 1s 超时轮询，超时时执行 drain 并打印 `[Subagent result]`）。
- Drain 时会把每个结果**直接追加**成一条 **assistant** 消息：`[subagent qa job_id=N] ...`。
- 因此 **context 里会出现连续多条 assistant**（例如 user → assistant → user → assistant → assistant → assistant），而**没有**中间的 user 消息。这就是「async fork 引入的变量」。

## 3. 现有功能是否受影响

- **发 API 的 `request_messages`**：就是 `ctx.messages`。OpenAI 兼容 API 允许多条连续 assistant（多轮对话历史），当前用法没问题。
- **Compaction**：只按「条」压缩，不假设 user/assistant 交替，不受影响。
- **/debug、/history**：只读 `ctx.messages` 并展示，连续 assistant 会正常显示。
- **严格交替的后端**：若将来接入**要求** user/assistant 严格交替的 API，需要在发请求前做一次 **normalize**（见下节），当前未做。

## 4. 若将来要「严格 user–assistant」可选方案

在**不改变**现有 drain 逻辑、不破坏其他功能的前提下，可加一层**发请求前的归一化**（例如在 `agent_loop` 里构建 `request_messages` 时）：

- **方案 A**：把连续多条的 `[subagent ...]` assistant **合并成一条** assistant，内容用明确分隔（如 `--- job_id=1 --- ... --- job_id=2 ---`），这样整段历史仍是 user/assistant 交替。
- **方案 B**：在每条「drain 出来的」assistant 前插一条**合成 user**（如 `[Subagent job_id=N result below]`），再跟一条 assistant。这样 context 变长，但严格交替。

两种都只在「发往该后端」的 `request_messages` 上做，**不**改 `ctx.messages` 的存储格式，这样 /debug、compaction、日志都保持现状。

## 5. 自动化测试应遵守的约定

- **不并发送请求**：每次只发一条，等 `>>> Ready for input.`（或等价条件）后再发下一条。
- **等 subagent 结果**：发完多个 `/fork` 后，不要立刻发下一条命令，而是**轮询 log**，直到出现足够多的 `[subagent qa job_id=...]`（或 `[Subagent result]`），再发 `/fork-wait`、`/fork-status`。这样既符合「一行一轮」的协议，又覆盖了 drain 注入的异步结果。
- **断言**：除 Ready 次数、无遗留线程、log 中 job_id 与 read_file 等外，可断言 context 中确实出现连续 assistant（若需验证 drain 行为）。

这样测试与「live session 严格逐行」和「async fork 导致连续 assistant」两者都兼容，且不要求改现有其他功能。
