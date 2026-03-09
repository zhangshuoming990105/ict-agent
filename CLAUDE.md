# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Run

```bash
pip install -e ".[dev]"
export KSYUN_API_KEY="your-key"

python main.py                              # workspace = cwd
python main.py --task task/example_axpby    # CUDA task context
python main.py --safe-shell                 # shell command approval
python main.py --sandbox                    # process-level sandbox
```

## Testing

```bash
python -m pytest tests/unit tests/integration_mock_api -v        # 64 tests, no API
ICT_AGENT_RUN_REAL_API=1 python -m pytest tests/integration_real_api -v  # 3 tests, needs API
python scripts/run_mixed_e2e.py -v                               # lightweight live e2e (3 turns)
```

### Live Session E2E Pattern

The agent runs as a background process via FIFO pipe. `scripts/live_session.sh` manages sessions under `.live_session/session_<ID>/`:

```python
# 1. Start
subprocess.Popen(["bash", "scripts/live_session.sh", "--session-id", "0", "start", "--model", "gpt-oss-120b"])

# 2. Wait for ready (polls log for ">>> Ready for input.")
wait_ready(log_path, min_count=1, timeout_sec=90)

# 3. Send + wait
live_send(session_id, "your message")
wait_ready(log_path, min_count=2, timeout_sec=90)  # each turn increments count by 1

# 4. Assert on log
assert "Calling tool: read_file" in log_path.read_text()

# 5. Quit
live_send(session_id, "quit")
```

Use `--model gpt-oss-120b` for tests. Session IDs: 0=live_e2e, 1=fork_smoke, 2=enhancements, 3=multi_fork, 4=mixed_e2e.

## Architecture

### Source (`src/ict_agent/`)

- **app/** — CLI (`cli.py`), bootstrap, config
- **runtime/agent_loop.py** — Core loop. Key mechanisms:
  - `start_async_streaming_call()` — streaming with incremental tool_call merging
  - `_maybe_persist_large_output()` — >30K char results saved to `.tool_outputs/` (workspace-relative so `read_file` can access)
  - `CORE_TOOLS` — 8 core tools (read_file, write_file, edit_file, run_shell, list_directory, search_files, grep_text, workspace_info); others injected on demand (fork tools on keyword match)
  - `MAX_TOKENS_TOOL_TURN=2048` / `MAX_TOKENS_FINAL_TURN=8192`
  - Preemption, recovery, auto-compaction
- **runtime/logging.py** — `RunLogger` with `print_streaming()` / `end_streaming()`
- **tools.py** — `@tool` registry + shell safety (`SAFE_COMMANDS`, `BANNED_COMMAND_PATTERNS`, wildcard allowlist/denylist)
- **utils/edit_diff.py** — `edit_file` surgical edit logic (fuzzy match, line endings, BOM)
- **sandbox.py** — bubblewrap (Linux) / seatbelt (macOS) process isolation
- **skills.py** — loads `skills/*/SKILL.md`, trigger-based selection, `inline`/`fork` modes
- **context.py** — `ContextManager`: messages, tiktoken counting, compaction
- **compactor.py** — LLM-based summarization
- **llm.py** — Provider abstraction (ksyun/infini/auto)
- **commands/** — `CommandRegistry` dispatches `/` commands
- **domains/cuda/** — `CudaDomainAdapter`, GPU selection, task management, recovery

### Skills

| Skill | Mode | Purpose |
|-------|------|---------|
| core | inline | Always-on (calculator, get_current_time, shell_policy_status) |
| filesystem | inline | File operations |
| shell | inline | Shell commands |
| cuda | inline | CUDA development |
| session | inline | Multi-session orchestration |
| scout | fork | Read-only codebase exploration |
| qa | fork | Q&A with tools |

### Slash Commands

`/help` `/tokens` `/history` `/debug` `/model` `/compact` `/skills` `/skill` `/run` `/fork` `/fork-status` `/fork-wait` `/verbose` `/shell-safe` `/shell-policy` `/preempt` `/recovery` `/workspace` `/clear`

## Conventions

- **Doc updates are mandatory.** After completing a feature or test change, always check and update the relevant docs (`CLAUDE.md`, `docs/testing.md`, `README.md`). Test counts, command examples, and architecture descriptions must stay in sync with the code.
- Python 3.10+. Source under `src/ict_agent/` with `setuptools` package-dir layout.
- OpenAI SDK against compatible endpoints. Streaming by default in the main chat loop.
- Workspace root = `cwd` (no `--task`) or `task_dir/workdir` (with `--task`).
- `.tool_outputs/` stores persisted large outputs (gitignored).
- `.shell_policy.json` stores user shell allowlist/denylist (gitignored).
- Tests use `gpt-oss-120b` (weaker model) for robustness.
- Real API tests gated by `ICT_AGENT_RUN_REAL_API=1`.
- See `docs/testing.md` for full roster, `docs/live_session.md` for session ops, `docs/fork_and_subagents.md` for fork details.
