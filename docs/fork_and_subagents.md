# Fork Skills & Subagents

Skills with `context: fork` in their SKILL.md frontmatter run as **subagents**: isolated context, dedicated tools, single task, result returned to the main conversation.

## User Commands

- `/run <skill> <task>` — run a fork skill **synchronously**; result injected when done
- `/fork <skill> [full_context] <task>` — start **asynchronously**; result injected next turn

## Agent Tools

- `fork_subagent(skill_name, task)` — start in background; returns `job_id`
- `get_subagent_result(job_id, timeout_sec)` — wait for result by `job_id`

## Parallel Usage

Start multiple forks; completed results are drained into the conversation at the start of each new turn.


---

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

---

# Multi-fork 测试说明

Fork skill（如 **qa**）以 subagent 运行：独立 context、专用工具，结果注入主对话。支持 **final**（仅最后回复）与 **full_context**（整段对话）两种返回方式。

---

## 返回格式与选项

- **job_id**：注入主 context 时带 `job_id`，便于对应。格式 `[subagent qa job_id=1] ...`（/fork）或 `job_id=0`（/run）。/debug 与终端均可看到。
- **return_mode**
  - **final**（默认）：只返回 subagent 最后一轮回复，省主 context。
  - **full_context**：返回整段 subagent 对话（user/assistant/tool 序列化），适合 handoff。
  - 用法：`/fork qa full_context <task>` 或工具参数 `return_mode: "full_context"`。

---

## 推荐测试流程（qa + read_file）

题目文件：`skills/qa/questions/q01.txt` … `q16.txt`。按顺序输入：

**1. 启动两个 subagent（各答 8 题）**

```
/fork qa Read skills/qa/questions/q01.txt through q08.txt with read_file, answer each, reply with numbered list 1-8.
```

```
/fork qa Read skills/qa/questions/q09.txt through q16.txt with read_file, answer each, reply with numbered list 9-16.
```

**2. 等两段 `[Subagent result]` 后，确认无遗留线程**

```
/fork-wait 90
```

```
/fork-status
```

期望：`No fork subagents running` 或 `all finished`。

**3. 测 full_context（可选）**

```
/fork qa full_context Read skills/qa/questions/q01.txt and q02.txt with read_file, answer each, reply with numbered list 1-2.
```

在 /debug 中查看 `[subagent qa job_id=1]` 内容应包含 `--- user ---`、`--- tool (read_file) ---`、`--- assistant ---` 等整段对话。

---

## 参考

- 题目列表：`tests/data/fork_quiz_questions.py` 或 `skills/qa/questions.txt`
- 自动化测试（与上述流程一致）：`python scripts/run_multi_fork_test.py [--session-id 0] [--model gpt-oss-120b]`；加 `-v`/`--verbose` 可打印每一步和 log 轮询进度。
- Session 与 async fork 的约束（顺序发送、drain 导致连续 assistant）：`docs/async_fork_and_session.md`
