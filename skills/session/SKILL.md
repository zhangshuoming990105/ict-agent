---
name: session
description: Multi-session orchestration skill for interacting with other live agent sessions.
tools:
  - run_shell
triggers:
  - session
  - multi-agent
  - other agent
  - agent0
  - agent1
  - session0
  - session1
  - live session
  - dialogue
  - talk to
  - ask
  - reply
  - output
  - read output
  - send message
  - 提问
  - 问题
  - 回复
  - 输出
  - 读取输出
  - 读取回复
  - 查看session
  - 其他agent
  - 多agent
  - 发送消息
  - 等回复
  - 等待回复
always_on: false
---

# Session Skill

Use this skill when the task involves multiple live agent sessions.

## Core idea

Each live agent runs in a separate `session_id`. If not specified, the default is `0`.
When this skill is active, treat session operations as first-class actions and use the helper scripts below through `run_shell`.

## Key scripts

### List all sessions

```bash
bash scripts/list_sessions.sh
```

This prints:
- `current_session_id`
- all discovered `session_id`s
- whether each one is running
- pid, fifo path, and log path

### Send a message to another session

```bash
bash scripts/live_session.sh --session-id 1 send "hello from session 0"
```

### Read another session's output

Read the recent raw log tail:

```bash
bash scripts/read_session_output.sh --session-id 1 --lines 80
```

Read only the latest assistant reply:

```bash
bash scripts/read_session_output.sh --session-id 1 --assistant-only
```

### Wait for a fresh reply after sending a message

Do not immediately read the old latest reply after a new `send`. Prefer:

```bash
LINES=$(wc -l < .live_session/session_1/stdout.log)
bash scripts/live_session.sh --session-id 1 send "hello"
bash scripts/wait_for_session_reply.sh --session-id 1 --after-lines "$LINES"
```

This avoids accidentally reading a stale assistant reply from before the new message.

### Close a session reliably

```bash
bash scripts/close_session.sh --session-id 1
```

This first sends `quit`, waits for graceful shutdown, and only then falls back to a forced stop.

## Recommended workflow for agent-to-agent interaction

1. Run `bash scripts/list_sessions.sh` to discover your own session and the target session.
2. Record the current log line count of the target session.
3. Send one message at a time to the target session.
4. Wait for a fresh assistant reply with `wait_for_session_reply.sh`.
5. Read the target session's new output before deciding the next message.
6. Do not assume the other session will answer in a fixed format.
7. If you are done, close the target session with `close_session.sh`.

## Important notes

- Your own session id is usually available as `current_session_id` from `list_sessions.sh`.
- Do not mix logs from different session ids.
- Prefer reading the target session's latest assistant output before sending the next turn.
- Do not close another session unless the user explicitly asked you to close it or the workflow clearly requires cleanup.
