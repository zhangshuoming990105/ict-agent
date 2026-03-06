# CUDA Codegen Task Template

This is the default task card for CUDA or HIP code generation tasks in `ict-agent`.

## Objective

- Input: a task directory containing `model.py`
- Goal: produce a correct and performant CUDA or HIP extension implementation
- Deliverables:
  - generated kernel sources in `kernels/`
  - binding code in `kernels/*_binding.cpp`
  - `model_new.py`
  - profiling evidence from `utils/profiling.py`

## Standard Workflow

1. Read and understand `model.py`
2. Implement kernels and bindings under `kernels/`
3. Write `model_new.py`
4. Compile with `bash utils/compile.sh`
5. Verify with `python -m utils.verification`
6. Profile with `python -m utils.profiling`

## Constraints

- Do not modify `binding.cpp`
- Do not modify `binding_registry.h`
- Do not modify `utils/`
