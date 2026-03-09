# ICT Agent

[![Tests](https://github.com/zhangshuoming990105/ict-agent/actions/workflows/test.yml/badge.svg)](https://github.com/zhangshuoming990105/ict-agent/actions/workflows/test.yml)
[![Tests (Real API)](https://github.com/zhangshuoming990105/ict-agent/actions/workflows/test-real-api.yml/badge.svg)](https://github.com/zhangshuoming990105/ict-agent/actions/workflows/test-real-api.yml)

A modular coding agent runtime with streaming responses, tool-calling, sandbox isolation, and CUDA task support. Uses Anthropic Messages API (with prompt caching) for Claude models and OpenAI-compatible APIs for others (ksyun/infini providers).

## Quick Start

```bash
pip install -e ".[dev]"
export KSYUN_API_KEY="your-key"       # or INFINI_API_KEY

ict-agent                             # general agent (workspace = cwd)
ict-agent --task task/example_axpby   # with CUDA task context
ict-agent --safe-shell                # enable shell command approval
ict-agent --sandbox                   # process-level sandbox (bubblewrap/seatbelt)
ict-agent "fix the bug"               # initial user message (positional)
ict-agent --append-system-prompt-file ./rules.txt   # append custom rules
cat prompt.txt | ict-agent            # stdin as initial message
```

The `ict-agent` command is installed via pip; use `python main.py` as fallback.

## Key Features

- **Streaming**: Real-time token-by-token output
- **Workspace = cwd**: Agent operates on the directory you launch from
- **Tool system**: `@tool` decorator with OpenAI function-calling format (file ops, shell, search, calculator, fork)
- **Skill system**: `skills/*/SKILL.md` with trigger-based activation and fork subagents
- **Token optimization**: Anthropic prompt caching (~90% input token savings), large output persistence (>30K → disk), dynamic tool schema pruning, per-turn max_tokens control
- **Sandbox**: Safe command whitelist, banned command blacklist, wildcard allowlist, bubblewrap/seatbelt process isolation
- **Context management**: tiktoken counting, auto-compaction, recovery cleanup

## Project Structure

```
ict-agent/
├── main.py                  # CLI entrypoint
├── src/ict_agent/
│   ├── app/                 # CLI, bootstrap, config
│   ├── runtime/             # agent_loop, session, preemption, logging
│   ├── commands/            # slash command registry
│   ├── domains/cuda/        # CUDA domain adapter
│   ├── tools.py             # tool registry + shell safety
│   ├── sandbox.py           # process-level isolation
│   ├── skills.py            # skill loader
│   ├── context.py           # ContextManager
│   ├── compactor.py         # context compaction
│   └── llm.py               # provider abstraction
├── skills/                  # skill definitions (SKILL.md)
├── tests/                   # unit + mock + real API tests
├── scripts/                 # live session management + e2e tests
└── docs/                    # detailed documentation
```

## Testing

```bash
python -m pytest tests/unit tests/integration_mock_api -v   # 64 tests, no API needed
ICT_AGENT_RUN_REAL_API=1 python -m pytest tests/integration_real_api -v  # needs API key
python scripts/run_enhancements_e2e.py -v                   # live agent e2e
```

See `docs/testing.md` for full test roster, live session testing patterns, and CI/CD details.

## Provider & Model Selection

```bash
ict-agent --list-providers                 # ksyun (default), infini, auto
ict-agent --list-models                     # list available models
ict-agent --provider ksyun --model mco-4    # specific provider/model
```

### Dual Provider Architecture

Claude models (`mco-4`, `mcs-1`) use the **Anthropic SDK** natively via the Messages API, enabling **prompt caching** — system prompt, tool definitions, and conversation history are cached across turns, reducing input token costs by ~90%. Other models (e.g. `deepseek-v3`) use the **OpenAI SDK** via Chat Completions API. The provider is selected automatically based on the model name.

Cache stats are displayed in turn usage:
```
[tokens: prompt=2,415, completion=4, total=2,489, cache_read=2,409, cache_write=70]
```

## Documentation

| Doc | Audience | Content |
|-----|----------|---------|
| `CLAUDE.md` | AI agents | Architecture, testing patterns, conventions |
| `docs/testing.md` | Developers | Test roster, CI/CD, live session e2e guide |
| `docs/live_session.md` | Developers | Session management, scripts reference |
| `docs/fork_and_subagents.md` | Developers | Fork skills, multi-fork, async patterns |

## (Optional) UniOpBench

```bash
cd benchmarks/UniOpBench && pip install -e .
```
