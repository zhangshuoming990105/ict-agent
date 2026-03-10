# UniOpBench Performance Report (K100)

- Run root: `/workspace/workspace/ict-agent/uniopbench/a100_uniopbench_v1/runs/20260308_222632`
- Python: `/usr/bin/python`
- Backend: `cuda`
- Generated at: `2026-03-10T14:47:29`
- Total artifacts: `48`
- Successful perf runs: `48`
- Failed runs: `0`

## Summary Statistics

| Metric | Value |
|--------|-------|
| Arithmetic Mean Speedup | **3.17x** |
| Geometric Mean Speedup | **1.09x** |
| Median Speedup | **1.14x** |
| Kernels < 0.8x | **15** / 48 |

## Per-Operator Results

| Operator | Status | Ref (ms) | Test (ms) | Speedup | Note |
|---|---:|---:|---:|---:|---|
| activation__elu | ok | 0.304320 | 0.294879 | 1.03x |  |
| activation__gelu | SLOW | 0.039359 | 0.058080 | 0.68x |  |
| activation__hardshrink | SLOW | 0.055680 | 0.079520 | 0.70x |  |
| activation__hardswish | ok | 0.034560 | 0.038240 | 0.90x |  |
| activation__relu | ok | 0.034880 | 0.036000 | 0.97x |  |
| activation__sigmoid | SLOW | 0.035680 | 0.077120 | 0.46x |  |
| activation__swish | SLOW | 0.056800 | 0.076320 | 0.74x |  |
| attention__gqa | ok | 1.225279 | 0.386080 | 3.17x |  |
| attention__merge_attn_states | ok | 0.173279 | 0.072000 | 2.41x |  |
| attention__mha | SLOW | 23.364624 | 107.781998 | 0.22x |  |
| attention__rope | ok | 0.131360 | 0.018080 | 7.27x |  |
| conv__conv1d | ok | 0.047360 | 0.014400 | 3.29x |  |
| conv__conv2d | SLOW | 0.065120 | 0.191840 | 0.34x |  |
| conv__depthwiseconv | ok | 0.213600 | 0.068480 | 3.12x |  |
| cvcuda__adaptive_threshold | ok | 40.817417 | 4.430238 | 9.21x |  |
| elementary__add | SLOW | 0.046560 | 0.060160 | 0.77x |  |
| elementary__sign | ok | 0.017440 | 0.013920 | 1.25x |  |
| elementary__sin | ok | 0.017280 | 0.013760 | 1.26x |  |
| elementary__square | ok | 0.013440 | 0.011360 | 1.18x |  |
| elementary__sub | ok | 0.018400 | 0.022080 | 0.83x |  |
| matmul__bmm | SLOW | 0.049280 | 1.893439 | 0.03x |  |
| matmul__dense | SLOW | 0.054080 | 2.981758 | 0.02x |  |
| matmul__dot-product | SLOW | 0.050560 | 0.485120 | 0.10x |  |
| matmul__gemm | SLOW | 0.129920 | 0.580640 | 0.22x | claude_code |
| matmul__gemv | SLOW | 0.020800 | 0.031360 | 0.66x |  |
| norm__batchnorm | ok | 0.028160 | 0.031520 | 0.89x |  |
| norm__instancenorm | ok | 0.045920 | 0.042400 | 1.08x |  |
| norm__layernorm | ok | 0.051680 | 0.040320 | 1.28x |  |
| norm__rmsnorm | ok | 0.077440 | 0.037440 | 2.07x |  |
| other__concat | ok | 0.068160 | 0.057440 | 1.19x |  |
| other__deformable | ok | 4.942236 | 0.328800 | 15.03x |  |
| other__embedding | ok | 0.029120 | 0.022240 | 1.31x |  |
| other__gatemlp | ok | 1.823678 | 0.029600 | 61.61x |  |
| other__gather | ok | 0.022400 | 0.021760 | 1.03x | claude_code |
| other__nms | SLOW | 0.360160 | 1.204960 | 0.30x |  |
| other__scatter | ok | 0.025920 | 0.018560 | 1.40x | claude_code |
| other__transpose | ok | 0.147040 | 0.076000 | 1.93x |  |
| pool__avgpool | ok | 0.056960 | 0.051680 | 1.10x |  |
| pool__maxpool | ok | 0.021120 | 0.017440 | 1.21x |  |
| pool__minpool | ok | 0.088480 | 0.036000 | 2.46x |  |
| pool__sumpool | ok | 0.398560 | 0.202560 | 1.97x |  |
| reduction__causal_softmax | ok | 0.045759 | 0.032800 | 1.40x | claude_code |
| reduction__histogram | ok | 0.135040 | 0.014720 | 9.17x |  |
| reduction__max | ok | 0.021439 | 0.014560 | 1.47x |  |
| reduction__mean | ok | 0.019680 | 0.014400 | 1.37x |  |
| reduction__min | SLOW | 0.020160 | 0.071040 | 0.28x |  |
| reduction__softmax | ok | 0.072960 | 0.069760 | 1.05x |  |
| reduction__sum | SLOW | 0.060800 | 0.080960 | 0.75x |  |

## Kernels Below 0.8x Speedup

| Operator | Speedup | Ref (ms) | Test (ms) |
|---|---:|---:|---:|
| matmul__dense | **0.02x** | 0.054080 | 2.981758 |
| matmul__bmm | **0.03x** | 0.049280 | 1.893439 |
| matmul__dot-product | **0.10x** | 0.050560 | 0.485120 |
| attention__mha | **0.22x** | 23.364624 | 107.781998 |
| matmul__gemm | **0.22x** | 0.129920 | 0.580640 |
| reduction__min | **0.28x** | 0.020160 | 0.071040 |
| other__nms | **0.30x** | 0.360160 | 1.204960 |
| conv__conv2d | **0.34x** | 0.065120 | 0.191840 |
| activation__sigmoid | **0.46x** | 0.035680 | 0.077120 |
| matmul__gemv | **0.66x** | 0.020800 | 0.031360 |
| activation__gelu | **0.68x** | 0.039359 | 0.058080 |
| activation__hardshrink | **0.70x** | 0.055680 | 0.079520 |
| activation__swish | **0.74x** | 0.056800 | 0.076320 |
| reduction__sum | **0.75x** | 0.060800 | 0.080960 |
| elementary__add | **0.77x** | 0.046560 | 0.060160 |
