# UniOpBench Performance Report

- Run root: `/workspace/ict-agent/uniopbench/a100_uniopbench_v1/runs/20260308_222632`
- Python: `/usr/local/bin/python`
- Backend: `cuda`
- Generated at: `2026-03-10T15:18:56`
- Total artifacts: `48`
- Successful perf runs: `44`
- Failed runs: `4`
- Arithmetic mean speedup: `4.90x`
- Kernels with speedup < 1: `attention__mha (0.55x), matmul__bmm (0.21x), matmul__dense (0.34x), matmul__gemm (0.24x)`

| Operator | Status | Ref (ms) | Test (ms) | Speedup | Input dtypes | Input shapes |
|---|---:|---:|---:|---:|---|---|
| activation__elu | ok | 0.143040 | 0.120520 | 1.19x | float16 | (2048, 2048, 3, 4) |
| activation__gelu | ok | 0.039240 | 0.026440 | 1.48x | float16 | (2048, 2048) |
| activation__hardshrink | ok | 0.040560 | 0.026440 | 1.53x | float32 | (2048, 2048) |
| activation__hardswish | ok | 0.039560 | 0.025560 | 1.55x | float16 | (2048, 2048) |
| activation__relu | ok | 0.041200 | 0.025920 | 1.59x | float16 | (2048, 2048) |
| activation__sigmoid | ok | 0.039720 | 0.026560 | 1.50x | float16 | (2048, 2048) |
| activation__swish | ok | 0.039360 | 0.026640 | 1.48x | float32 | (2048, 2048) |
| attention__gqa | ok | 3.227040 | 0.063600 | 50.74x | float16 | (1, 16, 16, 16) |
| attention__merge_attn_states | ok | 0.366720 | 0.027440 | 13.36x | float16, float32 | (128, 32, 128) |
| attention__mha | ok | 17.387199 | 31.393280 | 0.55x | float32 | (64, 2048, 12, 512) |
| attention__rope | ok | 0.295520 | 0.026560 | 11.13x | float32 | (2048, 128) |
| conv__conv1d | ok | 0.084640 | 0.026680 | 3.17x | float32 | (5,) |
| conv__conv2d | ok | 0.097400 | 0.059960 | 1.62x | float32 | (32, 64, 3, 3) |
| conv__depthwiseconv | ok | 0.118680 | 0.026560 | 4.47x | float32 | (1, 128, 126, 126) |
| cvcuda__adaptive_threshold | no_perf |  |  |  | uint8, float32 | (16, 1080, 1920) |
| elementary__add | ok | 0.040480 | 0.026760 | 1.51x | float16 | (2048, 2048) |
| elementary__sign | ok | 0.038400 | 0.026200 | 1.47x | float32 | (45, 25) |
| elementary__sin | ok | 0.038560 | 0.026760 | 1.44x | float32 | (7, 1, 6, 7) |
| elementary__square | ok | 0.039920 | 0.026320 | 1.52x | float32 | (256,) |
| elementary__sub | ok | 0.040600 | 0.026360 | 1.54x | float32 | (2, 16, 1024) |
| matmul__bmm | ok | 0.053160 | 0.254040 | 0.21x | float16 | (4, 512, 512) |
| matmul__dense | ok | 0.116880 | 0.342480 | 0.34x | float16, float32 | (64, 768) |
| matmul__dot-product | ok | 0.067040 | 0.031560 | 2.12x | - | (1024, 1024) |
| matmul__gemm | ok | 0.113320 | 0.481760 | 0.24x | float32 | (1024, 1024) |
| matmul__gemv | ok | 0.050960 | 0.027040 | 1.88x | float32 | (1024, 1) |
| norm__batchnorm | ok | 0.061600 | 0.039600 | 1.56x | float32 | (16, 3, 32, 32) |
| norm__instancenorm | ok | 0.121240 | 0.026800 | 4.52x | float32 | (1, 3, 224, 224) |
| norm__layernorm | ok | 0.055920 | 0.027840 | 2.01x | float16 | (4096, 512) |
| norm__rmsnorm | ok | 0.112720 | 0.027480 | 4.10x | float16 | (4096, 512) |
| other__concat | ok | 0.044760 | 0.027080 | 1.65x | float32 | (4, 64, 112, 112) |
| other__deformable | ok | 4.175720 | 0.078280 | 53.34x | - | (1, 200, 4096) |
| other__embedding | ok | 0.066440 | 0.026400 | 2.52x | int32, float16 | (2048, 512) |
| other__gatemlp | ok | 0.203400 | 0.027360 | 7.43x | float16 | (32, 128) |
| other__gather | exit=1 |  |  |  | float32, int64 | (50, 128, 4), (4,) |
| other__nms | exit=1 |  |  |  | float32 | (2048, 4) |
| other__scatter | exit=1 |  |  |  | float32, int32 | (8, 768, 1, 1) |
| other__transpose | ok | 0.053040 | 0.026560 | 2.00x | float32 | (1024, 1024) |
| pool__avgpool | ok | 0.038600 | 0.027000 | 1.43x | float32 | (4, 128, 26, 26) |
| pool__maxpool | ok | 0.044880 | 0.027040 | 1.66x | float32 | (4, 64, 2, 2) |
| pool__minpool | ok | 0.103840 | 0.026720 | 3.89x | float32 | (4, 192, 11, 11) |
| pool__sumpool | ok | 0.222080 | 0.082400 | 2.70x | float32 | (16, 64, 60, 60) |
| reduction__causal_softmax | ok | 0.112360 | 0.037080 | 3.03x | float32 | (8, 16) |
| reduction__histogram | ok | 0.178800 | 0.026840 | 6.66x | int32 | (1025,) |
| reduction__max | ok | 0.045320 | 0.025840 | 1.75x | float32 | (32, 32) |
| reduction__mean | ok | 0.045840 | 0.026160 | 1.75x | float32 | (64,) |
| reduction__min | ok | 0.046280 | 0.026000 | 1.78x | float32 | (64, 64) |
| reduction__softmax | ok | 0.039640 | 0.034320 | 1.16x | - | (4096, 1024) |
| reduction__sum | ok | 0.087360 | 0.027000 | 3.24x | float32 | (4096, 123) |

## Failures

### cvcuda__adaptive_threshold

- Artifact: `/workspace/ict-agent/uniopbench/a100_uniopbench_v1/runs/20260308_222632/operators/cvcuda__adaptive_threshold/artifact`
- Exit code: `0`
- Timed out: `False`
- Error summary:
```text
--------------------------------------------------------------------------------
SAMPLE OUTPUT (first 4 values):
PyTorch   : [0, 255, 255, 255]
CUDA      : [0, 255, 255, 255]
✅ STATUS: PASSED
================================================================================
/usr/local/lib/python3.12/site-packages/torch/cuda/__init__.py:61: FutureWarning: The pynvml package is deprecated. Please install nvidia-ml-py instead. If you did not install pynvml directly, please report this to the maintainers of the package that installed pynvml for you.
import pynvml  # type: ignore[import]
```

### other__gather

- Artifact: `/workspace/ict-agent/uniopbench/a100_uniopbench_v1/runs/20260308_222632/operators/other__gather/artifact`
- Exit code: `1`
- Timed out: `False`
- Error summary:
```text
================================================================================
TEST RESULT: CUDA vs PyTorch
================================================================================
ERROR: Failed to load kernel: CUDA Source File Not Found: /workspace/ict-agent/uniopbench/a100_uniopbench_v1/runs/20260308_222632/operators/other__gather/artifact/cuda_/kernel.cu
❌ STATUS: FAILED
================================================================================
/usr/local/lib/python3.12/site-packages/torch/cuda/__init__.py:61: FutureWarning: The pynvml package is deprecated. Please install nvidia-ml-py instead. If you did not install pynvml directly, please report this to the maintainers of the package that installed pynvml for you.
import pynvml  # type: ignore[import]
```

### other__nms

- Artifact: `/workspace/ict-agent/uniopbench/a100_uniopbench_v1/runs/20260308_222632/operators/other__nms/artifact`
- Exit code: `1`
- Timed out: `False`
- Error summary:
```text
Testing NMS Operator with CUDA
================================================================================
>> Running NMS: Boxes=2048, Thresh=0.5
🔧 Compiling CUDA kernel...
✅ CUDA kernel compiled successfully: /workspace/ict-agent/uniopbench/a100_uniopbench_v1/runs/20260308_222632/operators/other__nms/artifact/cuda_/lib_cuda_kernel.so
❌ FAILED
/usr/local/lib/python3.12/site-packages/torch/cuda/__init__.py:61: FutureWarning: The pynvml package is deprecated. Please install nvidia-ml-py instead. If you did not install pynvml directly, please report this to the maintainers of the package that installed pynvml for you.
import pynvml  # type: ignore[import]
```

### other__scatter

- Artifact: `/workspace/ict-agent/uniopbench/a100_uniopbench_v1/runs/20260308_222632/operators/other__scatter/artifact`
- Exit code: `1`
- Timed out: `False`
- Error summary:
```text
^^^^^^^^^^^^^^^^^^^^^^^^
File "/usr/local/lib/python3.12/site-packages/torch/cuda/__init__.py", line 1040, in synchronize
return torch._C._cuda_synchronize()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
RuntimeError: CUDA error: device-side assert triggered
CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.
For debugging consider passing CUDA_LAUNCH_BLOCKING=1
Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.
```
