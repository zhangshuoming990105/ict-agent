# ICT Agent

`ict-agent` is a staged refactor of `08_preemptible_cuda_agent`.

It keeps the Step 08 behavior baseline while reorganizing the code into clearer
modules:

- generic agent runtime
- slash command registry
- reusable tool system
- CUDA task domain adapter
- layered tests

## Default Behavior

`ict-agent` now starts in a **general-purpose agent mode** by default.

That means:

- the base system prompt is general coding / automation oriented
- CUDA-specific constraints are not applied unless a CUDA task is actually loaded
- specialized behaviors such as CUDA work or multi-session orchestration are expected to come from dynamically activated skills

In practice:

- `python main.py`
  starts a general agent
- `python main.py --task ...`
  starts the same general agent, but with CUDA task context layered on top

## Current Status

`ict-agent` is the new refactored codebase, but `08_preemptible_cuda_agent` is still
the baseline reference implementation.

On machines without CUDA or HIP support, focus on:

- project structure
- generic runtime behavior
- slash command routing
- mock and static validation

Defer CUDA execution and real GPU verification until compatible hardware is available.

## Quick Start

```bash
cd ict-agent
pip install -e ".[dev]"

export KSYUN_API_KEY="your-key"
# or
# export INFINI_API_KEY="your-key"

python main.py
python main.py --task task/example_axpby
```

## Provider And Model Selection

`ict-agent` now accepts both:

- `--provider`
- `--model`

Examples:

```bash
# show provider choices
python main.py --list-providers

# list models for the default provider (ksyun)
python main.py --list-models

# list models for a specific provider
python main.py --provider infini --list-models

# start with a specific provider/model pair
python main.py --provider ksyun --model mco-4
python main.py --provider infini --model deepseek-v3
```

Current provider behavior:

- default provider: `ksyun`
- optional provider: `infini`
- optional `auto` mode: prefer `ksyun`, fall back to `infini`

## Layout

```text
ict-agent/
├── main.py
├── agent_test.md
├── scripts/
├── src/ict_agent/
│   ├── app/
│   ├── commands/
│   ├── domains/cuda/
│   └── runtime/
└── tests/
```

## Test Layers

- `tests/unit`: pure logic
- `tests/integration_mock_api`: runtime flow with mocked model responses
- `tests/integration_real_api`: live-session style tests against real APIs

For the canonical live-session workflow, see `agent_test.md`.
For day-to-day session operations, cleanup, reset, and multi-session usage, see `LIVE_SESSION.md`.

Note:

- `scripts/reset_live_session.sh` resets only the live-session runtime state
- `scripts/clean_logs.sh` is the separate command for clearing persistent logs
