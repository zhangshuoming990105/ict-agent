# UniOpBench Task

This directory contains the repository-owned orchestration layer for running LLM kernel-generation experiments against the external [UniOpBench](../../benchmarks/UniOpBench) benchmark.

It does not modify the upstream UniOpBench source tree. Instead, it:
- reads an experiment config from `task.yaml`
- copies each selected operator into an isolated run workspace
- invokes the full `ict-agent` runtime inside that workspace
- lets the agent edit only `cuda_/kernel.cu`
- runs the copied operator's native `python test.py ...` commands
- stores prompts, agent traces, generated artifacts, and test logs under `task/task_results/uniopbench/<experiment.name>/runs/`

## Files

- [task.yaml](task.yaml): experiment config
- [TASK.md](TASK.md): task prompt injected into the agent runtime
- [OPTIMIZE_TASK.md](OPTIMIZE_TASK.md): task prompt for optimization rounds (perf-focused)
- [orchestrator.py](orchestrator.py): benchmark runner implementation
- [cli.py](cli.py): CLI argument parsing and entrypoint (invoked via `main.py --task uniopbench`)
- [session_runner.py](session_runner.py): live-session bridge that boots the normal `chat()` runtime for one operator workspace

## Config

`task.yaml` defines:
- `experiment`: model and sampling settings, repair budget, target CUDA arch, optional `max_agent_steps` (0 = unlimited)
- `operators`: explicit operator list, or `all`
- `prompt`: optional extra prompt fragments
- `runtime`: run-level behavior such as fail-fast and temp-build cleanup

Example:

```yaml
experiment:
  provider: auto
  model: deepseek-v3
  temperature: 0.2
  top_p: 0.95
  top_k: 50
  max_tokens: 8192
  max_repair_rounds: 3
  # max_agent_steps: 0   # optional; default = max(8, max_repair_rounds*4); 0 = unlimited
  cuda_arch: sm_80

operators:
  - activation/relu
  - conv/depthwiseconv
```

To run every operator:

```yaml
operators:
  - all
```

## Run

From the repository root:

```bash
ict-agent --task uniopbench
```

Useful variants:

```bash
ict-agent --task uniopbench --operators all --dry-run
ict-agent --task uniopbench --operators norm/rmsnorm
ict-agent --task uniopbench --run-id my_run
ict-agent --task uniopbench --resume --run-id my_run
```

## Optimize

The `optimize` subcommand is a self-contained workflow for iterative kernel performance improvement. It does **not** require a prior `run` — it can generate a kernel from scratch or start from a reference implementation.

```bash
# From scratch: first round generates the kernel, subsequent rounds optimize
ict-agent --task uniopbench --operators norm/rmsnorm optimize --rounds 3 --target-speedup 1.5

# With a reference kernel (copied as v0 baseline, not counted as a round)
ict-agent --task uniopbench --operators norm/rmsnorm optimize \
  --rounds 3 --target-speedup 1.5 \
  --ref-impl path/to/kernel.cu

# Dry run to inspect prompts without calling the LLM
ict-agent --task uniopbench --operators norm/rmsnorm optimize --rounds 2 --dry-run
```

### Optimize flow

For each operator:
1. Copies operator scaffold into a fresh `artifact/` directory
2. If `--ref-impl` is given, copies it as `v0_baseline.cu` (not counted as a round)
3. Runs up to `--rounds` agent sessions:
   - Round 1 without ref-impl: uses the generation prompt (`TASK.md`) to create the kernel from scratch
   - Subsequent rounds: uses the optimization prompt (`OPTIMIZE_TASK.md`)
4. Each round's kernel is saved as a versioned snapshot in `versions/`
5. If correctness breaks, the kernel is reverted to the last good version
6. Early-stops when `--target-speedup` is met
7. Selects the best-performing version and copies it back to `artifact/cuda_/kernel.cu`

### Kernel versioning

Each operator tracks versions in `versions/manifest.json`:

```json
{
  "versions": [
    {"version": "v0_baseline", "file": "v0_baseline.cu", "speedup": null, "source": "ref_impl"},
    {"version": "v1_opt", "file": "v1_opt.cu", "speedup": 1.34, "source": "round_1", "correctness": true}
  ],
  "best": "v1_opt",
  "best_speedup": 1.34,
  "target_speedup": 1.0
}
```

## Execution Model

Each operator runs in an **isolated live agent workflow**:
- **Workspace**: Each operator gets its own workspace (`artifact/` directory); no shared workspace across operators.
- **Context**: Each repair round starts a fresh live `chat()` session. No conversation history is carried across rounds; only the regenerated round request file carries prior kernel/log context forward.

For each operator, the orchestrator:
1. copies the upstream operator into a run-local `artifact/` directory
2. builds a round system prompt from `TASK.md` plus the operator scaffold
3. writes each round's detailed request into `artifact/.uniopbench/requests/round_<n>.md`
4. launches the normal `ict-agent` live `chat()` runtime for that round with the operator's artifact as the workspace root
5. sends one short user turn per repair round that tells the agent to read the corresponding request file
6. lets the agent inspect files, edit `cuda_/kernel.cu`, and run the operator's native test commands
7. if compile or correctness fails, writes a new repair request file with the prior kernel and logs, then starts a new live session for the next round

Success is currently defined as:
- `python test.py --compile-only` passes
- `python test.py --no-perf` passes
- if variants are supported, `python test.py --variants yaml --no-perf` passes

Performance is recorded, but it is not currently used as the pass/fail gate.

## Output Layout

Runs are written to:

- `task/task_results/uniopbench/<experiment.name>/runs/`

The `experiment.name` comes from `task.yaml` (e.g. `a100_uniopbench_v1`). Each run looks like:

```text
<experiment.name>/runs/<run_id>/
  run_summary.json
  console.log
  operators/
    activation__relu/
      prompt/
        system.txt
        user.txt
      rounds/
        round_0/              # run subcommand: round_0, round_1, ...
          agent.log           # optimize subcommand: round_1, round_2, ...
          trajectory.log
          request.json
          response.json
          extracted_kernel.cu
          verify.log
          perf.log
          variants.log
      versions/               # optimize only
        manifest.json
        v0_baseline.cu
        v1_initial.cu / v1_opt.cu
        v2_opt.cu
      artifact/
        ...
      result.json
```

## Notes

- This task layer assumes UniOpBench dependencies are already installed in the environment you use to run it.
- The orchestrator uses the repository's `ict-agent` runtime, not a direct one-shot LLM call.
- `enable_torch_compile_baseline` is currently recorded in metadata only; it is not injected into the external UniOpBench runner.
