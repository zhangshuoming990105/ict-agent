# Example Pattern: Two-Input AddCustom

## Reference Behavior

`z = x + y` with same shape and dtype.

## Key Mapping

- json op name: `AddCustom`
- tiling class: `AddCustomTilingData`
- host registration: `OP_ADD(AddCustom)`
- kernel entry: `add_custom(...)`
- extension call: `EXEC_NPU_CMD(aclnnAddCustom, x, y, z)`

## Minimal Validation Plan

1. Random tensor correctness test against PyTorch add.
2. Include small/large shapes and non-multiple tail sizes.
3. Confirm extension import and operator dispatch path.

## Common Failures

- Op name drift between host and JSON.
- Missing output dtype propagation in infer-dtype.
- Wrong queue count in two-input kernels.
