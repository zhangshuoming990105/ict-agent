# Elementwise Kernel Notes (Ascend C)

## Single-Input Ops

Typical ops: `tanh`, `sigmoid`, `relu`.

- Queues: one input queue + one output queue.
- Pipeline: `CopyIn`, compute intrinsic, `CopyOut`.
- Tile by core-local length; handle tail tile explicitly.

## Two-Input Ops

Typical ops: `add`, `mul`, `sub`.

- Queues: two input queues + one output queue.
- Keep queue buffer sizes consistent with dtype and tile size.
- Validate broadcasting policy explicitly (if unsupported, reject early).

## Error-Driven Fixing Priorities

1. Signature mismatch (json/host/kernel/ext names).
2. Tiling serialization or `blockDim` mistakes.
3. Queue/tensor dtype mismatch.
4. Tail handling and shape mismatch.
5. Performance-only issues after correctness passes.
