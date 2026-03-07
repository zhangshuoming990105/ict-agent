# ICT Agent

[![Tests](https://github.com/zhangshuoming990105/ict-agent/actions/workflows/test.yml/badge.svg)](https://github.com/zhangshuoming990105/ict-agent/actions/workflows/test.yml)
[![Tests (Real API)](https://github.com/zhangshuoming990105/ict-agent/actions/workflows/test-real-api.yml/badge.svg)](https://github.com/zhangshuoming990105/ict-agent/actions/workflows/test-real-api.yml)

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
- `tests/integration_real_api`: live-session style tests against real APIs (require API key; see [Real API tests](#real-api-tests-in-ci) below)

For the canonical live-session workflow, see `agent_test.md`.

### Real API tests (in CI)

The **Tests (Real API)** workflow runs only when:

- a push to `main` has a commit message containing **`[real-api]`** (e.g. `git commit -m "fix: xxx [real-api]"`), or
- you trigger it manually from the [Actions](https://github.com/zhangshuoming990105/ict-agent/actions) tab.

Configure `KSYUN_API_KEY` or `INFINI_API_KEY` in the repo’s **Settings → Secrets and variables → Actions** so the workflow can call the APIs. The badge above reflects the status of this workflow. See `docs/testing.md` for local runs and details.
For day-to-day session operations, cleanup, reset, and multi-session usage, see `LIVE_SESSION.md`.

Note:

- `scripts/reset_live_session.sh` resets only the live-session runtime state
- `scripts/clean_logs.sh` is the separate command for clearing persistent logs

## Fork skills (Agent as Skill)

Skills with `context: fork` in their frontmatter run as **subagents**: isolated context, dedicated tools, single task, result returned to the main conversation.

- **User commands**
  - `/run <skill> <task>` — run a fork skill **synchronously**; the result is injected when the subagent finishes.
  - `/fork <skill> <task>` — start a fork skill **asynchronously**; the result is injected at the start of the next turn (so you can start several in parallel).
- **Agent tools** (main agent can call these during a turn)
  - `fork_subagent(skill_name, task)` — start a subagent in the background; returns a `job_id`. Use for parallel tasks (e.g. run scout on path A and path B).
  - `get_subagent_result(job_id, timeout_sec)` — wait for a subagent result by `job_id` (optional; results are also auto-injected next turn).
- **Parallel usage**: Start multiple forks (via `/fork` or `fork_subagent`); completed results are drained into the conversation at the start of each new turn, or you can call `get_subagent_result` to wait within the same turn.

### Multi-fork QA test (2, 4, 8, 16 subagents)

16 questions are split across 2, 4, 8, or 16 subagents (qa fork skill) to verify the fork system scales and no threads are left behind.

**Recommended: run manually first.** See **`docs/multi_fork_test_manual.md`** for the exact prompts to type (and in what order). Once that flow works for you, the same steps can be codified into `scripts/run_multi_fork_test.py`.

- Questions: `tests/data/fork_quiz_questions.py`
- QA skill: `skills/qa/` (calculator + get_current_time)
