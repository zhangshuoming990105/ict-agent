# Ascend C Custom Op Project Structure

## Five Required Files

1. `op_name.json`  
   Defines op name, inputs/outputs, formats, and dtypes.
2. `op_host/*_tiling.h`  
   Shared tiling data structure used by host and device.
3. `op_host/*.cpp`  
   Tiling function, shape/dtype inference, and op registration.
4. `op_kernel/*.cpp`  
   AI Core kernel implementation (`CopyIn -> Compute -> CopyOut`).
5. `ext/csrc/op.cpp`  
   PyTorch binding and `EXEC_NPU_CMD` glue.

## Consistency Checklist

- Same op name across json + host register + tiling register + kernel symbol + extension binding.
- Input/output names in JSON match GM_ADDR ordering and host signatures.
- Tiling fields are POD types and serialized via `SaveToBuffer`.

## Build Loop

1. Generate skeleton (`msopgen` or project template).
2. Apply source patches.
3. Build operator.
4. Build extension.
5. Run correctness + performance test.
6. Use first blocking error as next-round input.
