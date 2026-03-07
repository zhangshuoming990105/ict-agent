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
