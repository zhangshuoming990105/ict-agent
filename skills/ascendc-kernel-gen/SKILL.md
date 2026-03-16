---
name: ascendc-kernel-gen
description: Generates and iterates Ascend C custom operator projects from PyTorch reference code, including op json, tiling definition, host-side registration, kernel implementation, and CppExtension binding, with compile/test feedback loops for correction. Use when users ask to create, debug, optimize, or validate CANN/Ascend 910B custom kernels, operator templates, or kernel generation workflows.
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

## Instructions

1. Read the PyTorch reference behavior and tensor constraints.
2. Generate or patch the 5 required files:
   - op definition json
   - tiling header
   - host op registration cpp
   - device kernel cpp
   - `CppExtension/csrc/op.cpp`
3. Build and test in a loop; use error output as the next-round input.
4. Keep naming and registration identifiers fully consistent across all files.
5. Keep commits clean: do not include build outputs or temporary artifacts.

## References

- Kernel generation docs live in `dev/kernel_gen_demo`.
