# Scripts Summary

An overview of all 16 files in the `scripts/` directory. These scripts collectively manage **live agent sessions** — starting, stopping, monitoring, testing, and cleaning up background Python agent processes that communicate via named pipes (FIFOs) and log files.

---

## Architecture Overview

The scripts revolve around a **live session** model:
- The agent (`main.py`) runs as a background process.
- Input is fed through a **named pipe** (FIFO) at `.live_session/session_<ID>/stdin.fifo`.
- Output is captured to a **log file** at `.live_session/session_<ID>/stdout.log`.
- A **FIFO keeper** process holds the pipe open to prevent EOF.
- Sessions support a **TTL** (time-to-live) that auto-closes them after a timeout.
- A **">>> Ready for input."** marker in the log signals that the agent has finished processing a turn.

---

## Core Session Management

### `live_session.sh` (290 lines) — **Central session controller**
The main entry point for all session operations. Supports five subcommands:
- **`start [agent args...]`** — Creates session directory, FIFO, FIFO keeper, launches `main.py` in the background with unbuffered output piped to the log, and starts a TTL timer.
- **`send "<message>"`** — Writes a message into the session's FIFO (requires a running session).
- **`status`** — Reports whether the session is running, its PID, wrapper PID, TTL PID, and file paths.
- **`stop`** — Sends "quit" via the FIFO, waits up to 5 seconds for graceful exit, then force-kills if needed. Cleans up keeper, TTL timer, and stale files.
- **`paths`** — Prints all file paths (FIFO, log, PID files) for the session.

Supports `--session-id ID` (default `0`) and `--ttl SEC` (default `3600`).

### `close_session.sh` (77 lines) — **Graceful session shutdown**
A higher-level close script:
1. Sends `"quit"` to the session via `live_session.sh send`.
2. Polls for up to `--timeout` seconds (default 15) for the process to exit.
3. Falls back to `live_session.sh stop` (force-kill) if the timeout is exceeded.

### `list_sessions.sh` (39 lines) — **Enumerate all sessions**
Lists all session directories under `.live_session/`, showing for each:
- Session ID, running status, PID, FIFO path, and log path.
Also prints the current `ICT_AGENT_SESSION_ID`.

---

## Cleanup & Reset

### `cleanup_live_session.sh` (121 lines) — **Selective session cleanup**
Cleans stale session metadata (PID files, FIFOs, empty directories). Options:
- `--session-id ID` — Clean only one session.
- `--stop-running` — Close running sessions before cleanup.
- `--remove-logs` — Also remove `stdout.log` files.
Handles legacy root-level state files as well.

### `reset_live_session.sh` (35 lines) — **Full reset to clean state**
Nuclear option: closes all running sessions, kills lingering processes, removes all session state directories and files under `.live_session/`. Returns the repo to a clean baseline.

### `kill_lingering_live_session_processes.sh` (46 lines) — **Force-kill orphan processes**
Finds orphaned `live_session.sh start` wrapper processes and their children via `ps`, then kills them (`SIGTERM`, then `SIGKILL` after 1 second).

### `clean_logs.sh` (46 lines) — **Remove log files**
- Default: removes only `logs/session_*` (session-specific logs).
- `--all`: removes everything under `logs/`.

---

## Output Reading & Monitoring

### `read_session_output.sh` (91 lines) — **Read session log output**
Reads the tail of a session's `stdout.log`. Supports:
- `--lines N` — Number of lines to tail (default 80).
- `--assistant-only` — Uses an embedded Python script to extract only the latest `Assistant:` response block, stopping at known delimiters (tool calls, token counts, ready prompts).

### `wait_for_session_reply.sh` (111 lines) — **Poll for new assistant reply**
Waits (with timeout) for a new assistant reply to appear in the session log. Supports:
- `--after-lines N` — Only consider replies that appear after line N in the log.
- `--timeout SEC` — Max wait time (default 60s).
Uses an embedded Python script to parse `Assistant:` blocks and prints the latest one found after the specified line.

---

## Test & E2E Scripts

### `test_basic_live_session.sh` (79 lines) — **Basic smoke test**
Starts a session, sends 3 simple messages (current time, identity question, arithmetic), waits for each reply, then quits. Validates that the session starts, responds, and shuts down correctly.

### `test_set_model.sh` (89 lines) — **Model switching test**
Tests the `/set-model` command:
1. Starts a session with `--model mco-4`.
2. Sends a message to remember the number 42.
3. Sends `/set-model gpt-oss-120b` to switch models mid-conversation.
4. Sends `/tokens` to verify the active model.
5. Asks a follow-up question (42 × 2 = 84) to verify context is preserved across model switches.
Validates log output for model switch confirmation, correct model name, and expected answers.

### `run_20_turns.sh` (81 lines) — **20-turn stress test**
Sends 20 diverse messages to a running session, exercising:
- Tool usage (`write_file`, `read_file`, `run_shell`, `calculator`)
- File creation (C programs, Python scripts)
- Compilation & execution
- Directory listing
- Math calculations

Also tests meta-commands: `/tokens`, `/compact high`, `/debug raw`. Reports skipped turns (timeouts).

### `run_live_e2e.sh` (87 lines) — **Full end-to-end integration test**
Orchestrates a complete E2E test:
1. Starts a session with a specified model (default `gpt-oss-120b`).
2. Runs `run_20_turns.sh`.
3. Validates that: ≥23 ready signals appeared, key tools (`write_file`, `read_file`, `run_shell`) were called, compaction was attempted, and `/debug raw` produced JSON output.
Reports pass/fail with failure count.

### `run_observed_session.sh` (166 lines) — **Interactive/file-driven session with live log streaming**
A versatile session runner that streams log output in real time:
- **Interactive mode** (default): prompts the user for messages at an `observer>` prompt.
- **File mode** (`--messages-file`): reads messages from a file, one per line.
- Supports `--keep-session` to leave the session running after the script exits.
- Passes extra args after `--` to the agent (e.g., `--model`, `--task`).

---

## Test Data Files

### `debug_test_messages.txt` (15 lines)
A sequence of 15 test prompts in Chinese, designed to test:
- Tool usage (`write_file`, `read_file`)
- Arithmetic and general knowledge
- Python concepts, recursion, HTTP/HTTPS, Big-O notation
- Model switching (`/model gpt-oss-120b`)
- Context memory (recalling previously read file content after model switch)

### `debug_test_short.txt` (5 lines)
A shorter 5-message test sequence:
1. Write a file with `SECRET=42`
2. Read it back
3. Python question
4. Switch model (`/model gpt-oss-120b`)
5. Test memory recall of the SECRET value

---

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
| `run_20_turns.sh` | 20-turn stress test | `--session-id` |
| `run_live_e2e.sh` | Full E2E integration test | `--session-id`, `--model` |
| `run_observed_session.sh` | Interactive/file-driven with live streaming | `--session-id`, `--messages-file`, `--keep-session` |
| `debug_test_messages.txt` | 15-prompt test data (Chinese) | — |
| `debug_test_short.txt` | 5-prompt short test data (Chinese) | — |
