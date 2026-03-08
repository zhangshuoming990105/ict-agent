# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Summary

ict-agent is a modular coding agent runtime with CUDA task domain support. It provides an interactive CLI agent that uses OpenAI-compatible LLM APIs (via ksyun or infini providers) with tool-calling, slash commands, skill system, and fork-based subagent execution.

## Build & Run

```bash
pip install -e ".[dev]"          # install with dev dependencies (pytest)

python main.py                              # general agent (workspace = cwd)
python main.py --task task/example_axpby    # with CUDA task context
python main.py --safe-shell                 # enable shell command approval
python main.py --sandbox                    # enable process-level sandbox (bubblewrap/seatbelt)
python main.py --provider ksyun --model mco-4
python main.py --list-providers             # show available providers
python main.py --list-models                # list models for current provider
```

Requires `KSYUN_API_KEY` or `INFINI_API_KEY` environment variable.

## Testing

### Quick Commands

```bash
python -m pytest tests/unit tests/integration_mock_api -v        # CI-equivalent: 49 tests, no API needed
python -m pytest tests/ -k "test_name" -v                        # single test by name
ICT_AGENT_RUN_REAL_API=1 python -m pytest tests/integration_real_api -v  # real API tests (needs API key)
```

### Test Layers

| Layer | Dir | Count | Needs API | Notes |
|-------|-----|-------|-----------|-------|
| Unit | `tests/unit/` | 48 | No | Pure logic. `test_enhancements.py` covers the 6 new features (28 tests) |
| Mock integration | `tests/integration_mock_api/` | 1 | No | `process_tool_calls` with faked response |
| Real API (pytest) | `tests/integration_real_api/` | 10 | Yes | Gated by `ICT_AGENT_RUN_REAL_API=1`. Uses `gpt-oss-120b` |
| Real API (live e2e) | `scripts/run_*.py` | — | Yes | Full agent session via `live_session.sh` |

### Live Session E2E Testing

The agent runs as a **background process** communicating via named pipe (FIFO). This is the canonical way to test the full agent end-to-end without interactive input.

**Architecture**: `scripts/live_session.sh` manages sessions under `.live_session/session_<ID>/`:
- `stdin.fifo` — named pipe for sending messages
- `stdout.log` — all agent output
- `pid` — agent process ID
- A "ready" signal (`>>> Ready for input.`) in the log indicates the agent finished a turn

**Pattern for writing a live e2e test** (see `scripts/run_enhancements_e2e.py` as template):

```python
# 1. Start session (background)
subprocess.Popen(["bash", "scripts/live_session.sh", "--session-id", "0", "start", "--model", "gpt-oss-120b"])

# 2. Wait for first ready signal
wait_ready(log_path, min_count=1, timeout_sec=90)  # polls log for ">>> Ready for input."

# 3. Send message and wait for response
live_send(session_id, "你的测试消息")               # writes to stdin.fifo
wait_ready(log_path, min_count=2, timeout_sec=90)   # count increments each turn

# 4. Assert on log content
log_text = log_path.read_text()
assert "Calling tool: read_file" in log_text         # verify tool was called
assert "[large-output]" in log_text                   # verify feature behavior

# 5. Quit
live_send(session_id, "quit")
```

**Key conventions**:
- `wait_ready(log_path, N)` waits until `>>> Ready for input.` appears N times (each turn increments by 1; initial startup = 1)
- Use `--model gpt-oss-120b` (weaker model) for test robustness — avoids rate limits on strong models
- Session ID default: 0 for `run_live_e2e.py`, 1 for `run_fork_smoke.py`, 2 for `run_enhancements_e2e.py`
- Always stop session after test: `live_session.sh --session-id N stop`

**Available e2e scripts**:

| Script | Tests | Session ID |
|--------|-------|------------|
| `run_live_e2e.py` | 5 turns: time, write, read, compile, run + /tokens | 0 |
| `run_fork_smoke.py` | /run scout + main agent uses result | 1 |
| `run_enhancements_e2e.py` | workspace, large output, safe-shell, dynamic schema | 2 |
| `run_multi_fork_test.py` | Multi-fork QA (2 subagents) | 3 |

Run any of them with `-v` for step-by-step progress:
```bash
python scripts/run_enhancements_e2e.py -v
```

See `docs/testing.md` for the full test roster, CI/CD setup, and troubleshooting.

## Architecture

### Source Layout (`src/ict_agent/`)

- **app/** — CLI entrypoint (`cli.py`), bootstrap wiring (`bootstrap.py`), config dataclass (`config.py`)
- **runtime/** — Core agent loop (`agent_loop.py`), context/session management, preemption handling, logging
- **commands/** — Slash command registry pattern: `CommandRegistry` dispatches `/` commands to handler functions
- **domains/cuda/** — CUDA-specific domain adapter, task management, GPU selection, recovery logic, prompts
- **tools.py** — `@tool` decorator-based tool registry. All tools (file ops, shell, search, calculator, fork) registered here. Also contains shell safety logic (SAFE_COMMANDS, BANNED_COMMAND_PATTERNS, wildcard allowlist/denylist)
- **sandbox.py** — Process-level sandbox: bubblewrap (Linux) / seatbelt (macOS). Restricts shell command file writes to workspace only
- **skills.py** — Skill loading from `skills/*/SKILL.md` (YAML frontmatter + markdown body), trigger-based selection
- **context.py** — `ContextManager`: message history, token counting (tiktoken), compaction support, usage tracking
- **compactor.py** — Context compaction via LLM summarization
- **llm.py** — Provider abstraction (ksyun/infini/auto), OpenAI client creation

### Key Design Patterns

**Tool System**: Tools are registered with `@tool(name, description, parameters)` decorator in `tools.py`. The `execute_tool()` function dispatches by name. Tool schemas are OpenAI function-calling format.

**Skill System**: Skills live in `skills/*/SKILL.md` with YAML frontmatter (`name`, `tools`, `triggers`, `always_on`, `context`). Two modes:
- `inline` — instructions injected into main conversation context
- `fork` — runs as isolated subagent via `/run` (sync) or `/fork` (async)

**Agent Loop** (`runtime/agent_loop.py`): Per-turn loop with up to `max_agent_steps` autonomous LLM/tool rounds. Key features:
- Streaming responses via `start_async_streaming_call()` for real-time output
- Large output persistence: tool results >30K chars saved to disk (`_maybe_persist_large_output`)
- Dynamic tool schema: `CORE_TOOLS` set, non-core tools injected on demand
- Per-turn `max_tokens` control: `MAX_TOKENS_TOOL_TURN=2048` / `MAX_TOKENS_FINAL_TURN=8192`
- Preemption, recovery from tool failures, and automatic context compaction

**Domain Adapter**: `CudaDomainAdapter` composes system prompts, manages task loading, GPU selection, and registers domain-specific slash commands. This is the extension point for domain-specific behavior.

**Fork Subagents**: Background threads running isolated `run_fork_skill()` with their own `ContextManager`. Results drain into the main conversation via `fork_result_queue`.

### Skills

| Skill | Mode | Purpose |
|-------|------|---------|
| core | inline | Always-on base capabilities |
| filesystem | inline | File operations guidance |
| shell | inline | Shell command guidance |
| cuda | inline | CUDA development workflow |
| session | inline | Multi-session orchestration |
| scout | fork | Read-only codebase exploration |
| qa | fork | Q&A with calculator + time tools |

### Slash Commands

`/help`, `/tokens`, `/history`, `/debug`, `/model`, `/compact`, `/skills`, `/skill`, `/run`, `/fork`, `/fork-status`, `/fork-wait`, `/verbose`, `/shell-safe`, `/shell-policy`, `/preempt`, `/recovery`, `/workspace`, `/clear`

## Conventions

- Python 3.11+ required. Source under `src/ict_agent/` with `setuptools` package-dir layout.
- LLM interaction uses the OpenAI SDK (`openai` package) against OpenAI-compatible endpoints.
- Shell commands run sandboxed under workspace root with optional safe-mode (allowlist/denylist in `.shell_policy.json`).
- GPU selection: `--gpu auto` acquires an available GPU; `--gpu <index>` pins to a specific device.
- Context window management: token counting via tiktoken (`cl100k_base`), auto-compaction when approaching limit.
- Real API tests are gated behind `ICT_AGENT_RUN_REAL_API=1` env var and `[real-api]` in commit message for CI.

## Benchmarks

`benchmarks/UniOpBench/` is a separate sub-project for CUDA operator benchmarking. It has its own `CLAUDE.md`, dependencies (`requirements.txt`), and `optest` framework. Install separately with `pip install -e .` from that directory.
