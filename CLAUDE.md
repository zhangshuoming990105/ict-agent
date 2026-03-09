# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Run

```bash
pip install -e ".[dev]"
export KSYUN_API_KEY="your-key"

ict-agent                                   # workspace = cwd (or python main.py)
ict-agent --task task/example_axpby         # CUDA task context
ict-agent --safe-shell                      # shell command approval
ict-agent --sandbox                         # process-level sandbox
```

The `ict-agent` command is registered via `[project.scripts]` in pyproject.toml; use `python main.py` if not installed.

### System Prompt & Input (Claude Code style)

| Flag | Description |
|------|--------------|
| `--system-prompt "..."` | Replace entire system prompt |
| `--system-prompt-file PATH` | Load system prompt from file (replaces default) |
| `--append-system-prompt "..."` | Append text to default system prompt |
| `--append-system-prompt-file PATH` | Append file contents to default system prompt |
| `--input "..."` / `-i` | Initial user message (first turn) |
| `--input-file PATH` | Load initial user message from file |
| `ict-agent "query"` | Positional: initial user message |

When stdin is piped (not a TTY), it is used as the initial user message: `cat prompt.txt | ict-agent`.

### Live session

`ict-agent start/send/status/stop/paths` manages live sessions natively in Python. Start creates fifo, keeper, runs agent with stdin from fifo; RunLogger tees to `.live_session/session_<id>/stdout.log` via ICT_AGENT_LIVE_LOG. Send messages via `ict-agent send <message>` — stdin is FIFO, not terminal. See `docs/live_session.md`.

### No truncate

`--no-truncate` disables all truncation: full system prompt at startup, full tool results (read_file, list_directory, search_files, grep_text), no large-output persist to `.tool_outputs/`, full /history output.

## Testing

```bash
python -m pytest tests/unit tests/integration_mock_api -v        # 74 tests, no API
ICT_AGENT_RUN_REAL_API=1 python -m pytest tests/integration_real_api -v  # 7 tests, needs API
python scripts/run_mixed_e2e.py -v                               # lightweight live e2e (3 turns)
```

### Live Session E2E Pattern

The agent runs as a background process with stdin from a FIFO pipe. `ict-agent start` creates the FIFO, keeper process, and launches the agent natively in Python. Session state lives under `.live_session/session_<ID>/`.

```python
# 1. Start (native Python — no bash script needed)
subprocess.Popen(["ict-agent", "start", "--session-id", "0", "--model", "gpt-oss-120b"])

# 2. Wait for ready (polls log for ">>> Ready for input.")
log_path = root / ".live_session" / "session_0" / "stdout.log"
wait_ready(log_path, min_count=1, timeout_sec=90)

# 3. Send + wait
subprocess.run(["ict-agent", "send", "--session-id", "0", "your message"])
wait_ready(log_path, min_count=2, timeout_sec=90)  # each turn increments count by 1

# 4. Assert on log
assert "Calling tool: read_file" in log_path.read_text()

# 5. Quit
subprocess.run(["ict-agent", "send", "--session-id", "0", "quit"])
```

Use `--model gpt-oss-120b` for tests. Session IDs: 0=live_e2e, 1=fork_smoke, 2=enhancements, 3=multi_fork, 4=mixed_e2e.

## Architecture

### Source (`src/ict_agent/`)

- **app/** — CLI (`cli.py`), bootstrap, config, live session (`live_session.py`)
- **runtime/agent_loop.py** — Core loop. Key mechanisms:
  - `chat()` — unified entry point for interactive, live session, and headless/batch modes (`headless=True`)
  - `_run_single_turn()` — shared agent step loop (skill selection, model call, tool execution, recovery, autonomy); used by both interactive and headless
  - Dual streaming: `start_anthropic_streaming_call()` (Claude, with prompt caching) / `start_async_streaming_call()` (OpenAI)
  - `_openai_messages_to_anthropic()` / `_openai_tools_to_anthropic()` — format conversion at API boundary
  - Prompt caching: `cache_control` on system prompt (single merged block for Bedrock’s max-4 limit), last tool def, last user message (~90% input token savings)
  - `_maybe_persist_large_output()` — >30K char results saved to `.tool_outputs/`
  - `CORE_TOOLS` — 8 core tools; others injected on demand (fork tools on keyword match)
  - `MAX_TOKENS_TOOL_TURN=2048` (interactive) / `MAX_TOKENS_FINAL_TURN=8192` (always in headless)
  - Preemption, recovery, auto-compaction
- **runtime/logging.py** — `RunLogger` with `print_streaming()` / `end_streaming()`
- **tools.py** — `@tool` registry + shell safety (`SAFE_COMMANDS`, `BANNED_COMMAND_PATTERNS`, wildcard allowlist/denylist)
- **utils/edit_diff.py** — `edit_file` surgical edit logic (fuzzy match, line endings, BOM)
- **sandbox.py** — bubblewrap (Linux) / seatbelt (macOS) process isolation
- **skills.py** — loads `skills/*/SKILL.md`, trigger-based selection, `inline`/`fork` modes
- **context.py** — `ContextManager`: messages, tiktoken counting, compaction
- **compactor.py** — LLM-based summarization
- **llm.py** — `ModelRouter` dispatches by model name: Claude (`mco-4`, `mcs-1`) → Anthropic SDK; others → OpenAI SDK. `get_client_for_model()`, `is_anthropic_model()`, `/model` switching
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
- Dual SDK: Anthropic SDK (Messages API) for Claude models with prompt caching; OpenAI SDK for others. Streaming by default.
- Workspace root = `cwd` (no `--task`) or `task_dir/workdir` (with `--task`).
- `.tool_outputs/` stores persisted large outputs (gitignored).
- `.shell_policy.json` stores user shell allowlist/denylist (gitignored).
- Tests use `gpt-oss-120b` (weaker model) for robustness.
- Real API tests gated by `ICT_AGENT_RUN_REAL_API=1`.
- See `docs/testing.md` for full roster, `docs/live_session.md` for session ops, `docs/fork_and_subagents.md` for fork details.
