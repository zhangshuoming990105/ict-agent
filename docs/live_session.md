# Live Session Guide

这份文档专门说明 `ict-agent` 的 live session 机制，包括：

- 如何启动和关闭一个 session
- 如何清理 `.live_session`
- 如何重置整个 live-session 测试现场
- 如何并行运行多个 session
- 如何让其他 agent 作为外部 user 驱动主 agent

## 核心概念

一个 live session 就是一个长期运行的主 agent 进程，外部通过 FIFO 向它注入输入。

默认情况下，这个主 agent 是一个 **general-purpose agent**，不是一上来就进入 CUDA kernel 开发模式。

只有在以下情况成立时，才会叠加更专项的行为约束：

- 你显式加载了 CUDA task
- 当前输入触发了 `cuda` skill
- 当前输入触发了 `session` skill

所以推荐的理解方式是：

- 默认 base prompt 保持通用
- CUDA 是 domain/task mode
- 多 agent session 协作也是动态 skill mode

每个 session 都有一个 `session_id`：

- 默认是 `0`
- 你可以显式指定 `1`、`2`、`7` 等

每个 session 还有一个自动关闭 TTL：

- 默认 `ttl=3600` 秒
- 可以在启动时通过 `--ttl <秒数>` 覆盖
- 设为 `0` 或负数可关闭自动 TTL

不同 `session_id` 的状态是隔离的：

- `.live_session/session_<id>/stdin.fifo`
- `.live_session/session_<id>/stdout.log`
- `.live_session/session_<id>/pid`
- `logs/session_<id>/<timestamp>.log`

**重要**：FIFO、pid 等管道和状态文件**只有**通过 `ict-agent start` 子命令启动时才会被创建。若直接运行 `ict-agent` 或 `ict-agent --session-id 0`，CLI 只会用 `session_id` 写 `logs/session_<id>/` 下的日志，**不会**创建或使用 `.live_session/` 下的任何内容。所以若发现 session_0 里没有 stdin.fifo、pid 等，说明当前进程不是用 `ict-agent start` 起的，需要先停掉再用 `ict-agent start` 启动。

## 最常用命令

Live session 通过 `ict-agent` 子命令管理，全部用 Python 原生实现（不依赖 bash 脚本）：

```bash
ict-agent start  [--session-id ID] [--ttl SEC] [agent 参数...]   # 创建 FIFO + keeper，启动 agent
ict-agent send   [--session-id ID] "消息"                         # 向 FIFO 注入消息
ict-agent status [--session-id ID]                                # 查看是否运行中
ict-agent stop   [--session-id ID]                                # 优雅关闭
ict-agent paths  [--session-id ID]                                # 显示 fifo/log/pid 路径
```

`scripts/live_session.sh` 仍可用作备选，但 `ict-agent start` 是推荐入口。

### 启动

```bash
cd ict-agent
ict-agent start                                    # 默认 session_id=0, ttl=3600
ict-agent start --session-id 1                     # 指定 session
ict-agent start --session-id 1 --ttl 120           # 指定 TTL
ict-agent start --session-id 1 --model gpt-oss-120b  # 额外 agent 参数直接传递
```

### 发送消息

**重要**：`ict-agent start` 启动后，agent 的 stdin 是 FIFO，不是终端。在终端里直接打字不会传给 agent，必须用 `ict-agent send` 发送：

```bash
ict-agent send "现在几点？"
ict-agent send --session-id 1 "你是谁？"
```

### 查看状态

```bash
ict-agent status
ict-agent status --session-id 1
```

### 查看路径

```bash
ict-agent paths
ict-agent paths --session-id 1
```

## 关闭一个 Session

### 推荐方式

```bash
ict-agent stop --session-id 0
```

`ict-agent stop` 会：

1. 先通过 FIFO 发 `quit`
2. 等待 5 秒让 agent 正常退出
3. 若超时仍未退出，SIGKILL 强杀
4. 清理 keeper、TTL timer、stale pid 文件

也可以用 `bash scripts/close_session.sh --session-id 0`（先发 quit，再回退到 stop）。

## 清理 `.live_session`

### 只清状态，保留日志

```bash
bash scripts/cleanup_live_session.sh
```

默认行为：

- 清掉 stopped session 的 stale 状态文件
- 保留 `stdout.log`
- 默认不关闭正在运行的 session

### 清理前先关掉还在运行的 session

```bash
bash scripts/cleanup_live_session.sh --stop-running
```

### 连 stopped session 的 live log 一起删掉

```bash
bash scripts/cleanup_live_session.sh --remove-logs
```

### 两者一起

```bash
bash scripts/cleanup_live_session.sh --stop-running --remove-logs
```

## 清理持久化日志

### 只清 live-session 生成的持久化日志

```bash
bash scripts/clean_logs.sh
```

这会删除：

- `logs/session_*`

但不会影响其他非 session 的日志文件。

### 清空整个 `logs/`

```bash
bash scripts/clean_logs.sh --all
```

## 一键重置整个现场

```bash
bash scripts/reset_live_session.sh
```

这个脚本会：

1. 关闭所有仍在运行的 live sessions
2. 清空 `.live_session` 下的运行时状态

适合在每次新一轮多 agent / live-session 测试前执行。

## 多 Session

### 并行启动两个主 agent

```bash
ict-agent start --session-id 0
ict-agent start --session-id 1
```

### 分别给它们发消息

```bash
ict-agent send --session-id 0 "你扮演一个用户。"
ict-agent send --session-id 1 "你扮演一个助手。"
```

### 列出当前所有 session

```bash
bash scripts/list_sessions.sh
```

它会告诉你：

- 当前调用方默认视角里的 `current_session_id`
- 所有已发现的 session
- 每个 session 是否在运行
- pid、fifo、log 路径

## 动态 Skill

在多 session / 多 agent 场景里，最重要的两个专项 skill 是：

- `cuda`
- `session`

这两个 skill 都应该是动态注入的，而不是默认永远主导整个 agent。

也就是说：

- 普通问题：按 general agent 处理
- kernel / GPU / compile / verify / profile：激活 `cuda`
- session / 其他 agent / 读写其他 session 输出：激活 `session`

## 读取其他 Session 的输出

看最近若干行：

```bash
bash scripts/read_session_output.sh --session-id 1 --lines 80
```

只看最新 assistant 回复：

```bash
bash scripts/read_session_output.sh --session-id 1 --assistant-only
```

这非常适合“另一个 agent 作为 user 驱动主 agent”的场景。

## 适合其他 Agent 的工作流

如果你让另一个 agent 来驱动主 agent，推荐流程是：

1. `ict-agent status --session-id X` 或 `bash scripts/list_sessions.sh` 确认目标 session
2. `ict-agent send --session-id X "..."` 发消息
3. 读取 `.live_session/session_X/stdout.log` 或 `bash scripts/read_session_output.sh --session-id X --assistant-only` 读回复
4. 轮询 `">>> Ready for input."` 判断 turn 完成
5. 决定下一轮继续问什么

## 相关脚本速查

- `scripts/live_session.sh`
  用于 `start / send / status / stop / paths`
- `scripts/close_session.sh`
  可靠关闭某个 session
- `scripts/cleanup_live_session.sh`
  清 stale 状态，按需清 log
- `scripts/reset_live_session.sh`
  一键恢复到干净 live-session 运行现场（不删持久化 logs）
- `scripts/clean_logs.sh`
  清理 `logs/session_*` 或整个 `logs/`
- `scripts/list_sessions.sh`
  查看当前所有 session
- `scripts/read_session_output.sh`
  读取某个 session 的输出
- `scripts/run_observed_session.sh`
  以“观察者”方式逐轮驱动一个 session

## 建议

如果你准备开始一轮新的复杂测试，建议先执行：

```bash
bash scripts/reset_live_session.sh
```

这样最不容易被旧状态干扰。


---

# Scripts Reference

## Quick Reference

| Script | Purpose | Key Options |
|--------|---------|-------------|
| `live_session.sh` | Core session controller (start/send/status/stop/paths) | `--session-id`, `--ttl` |
| `close_session.sh` | Graceful shutdown with timeout fallback | `--session-id`, `--timeout` |
| `list_sessions.sh` | List all sessions and their status | — |
| `cleanup_live_session.sh` | Clean stale session metadata | `--session-id`, `--stop-running`, `--remove-logs` |
| `reset_live_session.sh` | Full reset of all session state | — |
| `kill_lingering_live_session_processes.sh` | Force-kill orphan processes | — |
| `clean_logs.sh` | Remove log files | `--all` |
| `read_session_output.sh` | Read/tail session log | `--session-id`, `--lines`, `--assistant-only` |
| `wait_for_session_reply.sh` | Poll for new assistant reply | `--session-id`, `--after-lines`, `--timeout` |
| `test_basic_live_session.sh` | 3-message smoke test | `--session-id` |
| `test_set_model.sh` | Model switching + context test | `--session-id` |
| `run_multi_fork_test.py` | Multi-fork QA test (2 subagents) | `--session-id`, `--model`, `-v` |
| `run_fork_smoke.py` | Fork smoke (/run scout + context) | `--session-id`, `--model`, `-v` |
| `run_live_e2e.py` | Live e2e (20 turns + compact + debug) | `--session-id`, `--model`, `-v` |
| `run_20_turns.sh` | 20-turn stress test | `--session-id` |
| `run_fork_smoke.sh` | Fork smoke (bash) | `--session-id`, `--model` |
| `run_live_e2e.sh` | Full E2E (bash) | `--session-id`, `--model` |
| `run_observed_session.sh` | Interactive/file-driven with live streaming | `--session-id`, `--messages-file`, `--keep-session` |
| `debug_test_messages.txt` | 15-prompt test data (Chinese) | — |
| `debug_test_short.txt` | 5-prompt short test data (Chinese) | — |
