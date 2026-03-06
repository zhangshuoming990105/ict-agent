---
name: cuda
description: CUDA kernel development and optimization skill for PyTorch extension work.
tools:
  - run_shell
  - read_file
  - write_file
  - list_directory
  - search_files
  - grep_text
triggers:
  - cuda
  - kernel
  - optimize
  - gpu
  - compile
  - profile
  - verify
  - model
  - performance
  - speedup
always_on: false
---

# CUDA Kernel Development Skill

You are working on PyTorch CUDA or HIP extension tasks.

Rules:
- Treat `model.py` as the source of truth
- Work in `kernels/` and `model_new.py`
- Do not modify `utils/`, `binding.cpp`, or `binding_registry.h`
- Use the standard compile, verify, and profile loop
- Prefer direct fixes from shell output instead of asking the user for procedural confirmation
