# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Summary

ict-agent is a modular coding agent runtime with CUDA task domain support. It provides an interactive CLI agent that uses OpenAI-compatible LLM APIs (via ksyun or infini providers) with tool-calling, slash commands, skill system, and fork-based subagent execution.

## Build & Run

```bash
pip install -e ".[dev]"          # install with dev dependencies (pytest)

python main.py                   # general-purpose agent mode
python main.py --task task/example_axpby  # with CUDA task context
python main.py --provider ksyun --model mco-4
python main.py --list-providers  # show available providers
python main.py --list-models     # list models for current provider
```

Requires `KSYUN_API_KEY` or `INFINI_API_KEY` environment variable.

## Testing

```bash
python -m pytest tests/ -v --tb=short           # all tests (unit + mock integration)
python -m pytest tests/unit -v                   # unit tests only
python -m pytest tests/integration_mock_api -v   # integration with mocked API
python -m pytest tests/integration_real_api -v   # real API (needs ICT_AGENT_RUN_REAL_API=1 + API key)
python -m pytest tests/unit/test_tools.py -v     # single test file
python -m pytest tests/ -k "test_name" -v        # single test by name
```

Test markers: `@pytest.mark.real_api` for live API tests. CI runs on Python 3.11 and 3.12.

## Architecture

### Source Layout (`src/ict_agent/`)

- **app/** — CLI entrypoint (`cli.py`), bootstrap wiring (`bootstrap.py`), config dataclass (`config.py`)
- **runtime/** — Core agent loop (`agent_loop.py`), context/session management, preemption handling, logging
- **commands/** — Slash command registry pattern: `CommandRegistry` dispatches `/` commands to handler functions
- **domains/cuda/** — CUDA-specific domain adapter, task management, GPU selection, recovery logic, prompts
- **tools.py** — `@tool` decorator-based tool registry. All tools (file ops, shell, search, calculator, fork) registered here
- **skills.py** — Skill loading from `skills/*/SKILL.md` (YAML frontmatter + markdown body), trigger-based selection
- **context.py** — `ContextManager`: message history, token counting (tiktoken), compaction support, usage tracking
- **compactor.py** — Context compaction via LLM summarization
- **llm.py** — Provider abstraction (ksyun/infini/auto), OpenAI client creation

### Key Design Patterns

**Tool System**: Tools are registered with `@tool(name, description, parameters)` decorator in `tools.py`. The `execute_tool()` function dispatches by name. Tool schemas are OpenAI function-calling format.

**Skill System**: Skills live in `skills/*/SKILL.md` with YAML frontmatter (`name`, `tools`, `triggers`, `always_on`, `context`). Two modes:
- `inline` — instructions injected into main conversation context
- `fork` — runs as isolated subagent via `/run` (sync) or `/fork` (async)

**Agent Loop** (`runtime/agent_loop.py`): Per-turn loop with up to `max_agent_steps` autonomous LLM/tool rounds. Supports preemption (user input during autonomous execution), recovery from tool failures, and automatic context compaction.

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
