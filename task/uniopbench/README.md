# UniOpBench Task

This directory contains the repository-owned orchestration layer for running LLM kernel-generation experiments against the external [UniOpBench](/Users/qiuchu/Project/ict-agent/benchmarks/UniOpBench) benchmark.

It does not modify the upstream UniOpBench source tree. Instead, it:
- reads an experiment config from `task.yaml`
- copies each selected operator into an isolated run workspace
- invokes the full `ict-agent` runtime inside that workspace
- lets the agent edit only `cuda_/kernel.cu`
- runs the copied operator's native `python test.py ...` commands
- stores prompts, agent traces, generated artifacts, and test logs under `task/task_results/uniopbench/<experiment.name>/runs/`

## Files

- [task.yaml](/Users/qiuchu/Project/ict-agent/task/uniopbench/task.yaml): experiment config
- [TASK.md](/Users/qiuchu/Project/ict-agent/task/uniopbench/TASK.md): task prompt injected into the agent runtime
- [orchestrator.py](/Users/qiuchu/Project/ict-agent/task/uniopbench/orchestrator.py): benchmark runner implementation

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
python test.py --task uniopbench
```

Useful variants:

```bash
python test.py --task uniopbench --dry-run
python test.py --task uniopbench --operators all
python test.py --task uniopbench --run-id my_run
python test.py --task uniopbench --resume --run-id my_run
python test.py --task uniopbench --config /Users/qiuchu/Project/ict-agent/task/uniopbench/task.yaml
```

## Execution Model

For each operator, the orchestrator:
1. copies the upstream operator into a run-local `artifact/` directory
2. builds a task prompt from `TASK.md` plus the operator's `test.py`, `torch_/ref.py`, `cases.yaml`, and optional legacy helper files
3. launches a single autonomous `ict-agent` batch turn in that copied workspace
4. lets the agent inspect files, edit `cuda_/kernel.cu`, and run the operator's native test commands
5. if compile or correctness fails, feeds the prior round's kernel and logs into the next repair round

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
  run_metadata.yaml
  run_summary.json
  console.log
  operators/
    activation__relu/
      metadata.yaml
      prompt/
        system.txt
        user.txt
      rounds/
        round_0/
          agent.log
          request.json
          response.json
          extracted_kernel.cu
          compile.log
          verify.log
          perf.log
          variants.log
      artifact/
        ...
      result.json
```

## Notes

- This task layer assumes UniOpBench dependencies are already installed in the environment you use to run it.
- The orchestrator uses the repository's `ict-agent` runtime, not a direct one-shot LLM call.
- `enable_torch_compile_baseline` is currently recorded in metadata only; it is not injected into the external UniOpBench runner.
