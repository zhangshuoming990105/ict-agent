---
name: ascendc-kernel-gen
description: Designs, generates, and iteratively fixes Ascend C custom operators from PyTorch reference behavior, covering op schema, tiling, host registration, kernel implementation, CppExtension glue, and compile/test feedback loops. Use when users ask to build, debug, validate, or optimize CANN/Ascend 910B custom op workflows, kernel templates, or operator code generation.
tools:
  - run_shell
  - read_file
  - write_file
  - list_directory
  - search_files
  - grep_text
triggers:
  - ascendc
  - ascend c
  - cann
  - 910b
  - custom op
  - kernel gen
  - op tiling
  - msopgen
  - cpp extension
always_on: false
---

# Ascend C Kernel Generation

## When To Apply

Apply this skill when requests mention Ascend C, CANN, custom op, tiling, 910B, `msopgen`, or PyTorch-to-Ascend kernel translation.

## Quick Start

1. Parse reference behavior (dtype, shape rules, edge cases, numerics).
2. Generate or patch the 5 required files for one operator:
   - `op_name.json`
   - `op_host/*_tiling.h`
   - `op_host/*.cpp`
   - `op_kernel/*.cpp`
   - `ext/csrc/op.cpp` (or `CppExtension/csrc/op.cpp`)
3. Build and run validation.
4. Feed compiler/runtime/test errors back into the next patch round.

## Required Guardrails

- Keep identifiers consistent across JSON, `REGISTER_TILING_DATA_CLASS`, `OP_ADD`, kernel entry name, and `EXEC_NPU_CMD`.
- Keep host infer-shape / infer-dtype logic aligned with PyTorch behavior.
- Prefer minimal patches per retry round so regressions are attributable.
- Never commit generated artifacts (`build`, `build_out`, round outputs, temp files).

## Output Contract

- Return a concise change summary.
- Report build/test status and first blocking error.
- If incomplete, state next concrete retry action.

## Additional Resources

- Operator structure reference: [project-structure.md](project-structure.md)
- Elementwise kernel patterns: [elementwise-kernel.md](elementwise-kernel.md)
- Two-input add example: [example-add.md](example-add.md)
- Source prompt assets in repo:
  - `dev/kernel_gen_demo/skills/L0_project_structure.md`
  - `dev/kernel_gen_demo/skills/L1_elementwise_kernel.md`
  - `dev/kernel_gen_demo/skills/L2_example_add.md`
