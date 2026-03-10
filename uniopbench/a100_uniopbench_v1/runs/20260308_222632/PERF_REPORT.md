# Performance Benchmark Report: Kernel Speedup vs PyTorch Baseline

**Run:** `a100_uniopbench_v1` / `20260308_222632`  
**Data Source:** Extracted from existing `perf.log` files (single test config per operator)  
**Baseline:** PyTorch reference implementation

> **Note:** For operators with `supports_variants=true`, the perf.log contains results from a single test configuration. Average speedup across all variants would require re-running with `--variants yaml` and `enable_perf` in the test suite.

---

## Summary

| Metric | Value |
|--------|-------|
| Operators with perf data | 34 / 35 |
| Operators with speedup > 1x | 26 / 34 |
| Arithmetic mean speedup | 3.42x |
| Missing data | `cvcuda__adaptive_threshold` (no PERFORMANCE section in perf.log) |

---

## Per-Operator Speedup

| Operator | Speedup | vs Baseline | Variants |
|----------|---------|-------------|----------|
| **Activation** | | | |
| activation__elu | **1.38x** | Faster | single |
| activation__gelu | **1.23x** | Faster | single |
| activation__hardshrink | 0.76x | Slower | single |
| activation__hardswish | **1.38x** | Faster | single |
| activation__relu | **1.50x** | Faster | single |
| activation__sigmoid | **1.44x** | Faster | single |
| activation__swish | **1.10x** | Faster | single |
| **Attention** | | | |
| attention__gqa | **9.99x** | Faster | single |
| attention__merge_attn_states | **13.25x** | Faster | single |
| attention__mha | 0.59x | Slower | single |
| attention__rope | **14.17x** | Faster | single |
| **Conv** | | | |
| conv__conv1d | **3.50x** | Faster | single |
| conv__conv2d | 0.42x | Slower | single |
| conv__depthwiseconv | **2.69x** | Faster | single |
| **CVCUDA** | | | |
| cvcuda__adaptive_threshold | N/A | - | (no perf in log) |
| **Elementary** | | | |
| elementary__add | **1.40x** | Faster | single |
| elementary__sign | **1.42x** | Faster | single |
| elementary__sin | **1.42x** | Faster | single |
| elementary__square | **1.42x** | Faster | single |
| elementary__sub | **1.50x** | Faster | single |
| **Matmul** | | | |
| matmul__bmm | 0.11x | Slower | single |
| matmul__dense | 0.20x | Slower | single |
| matmul__dot-product | **2.07x** | Faster | single |
| matmul__gemm | 0.29x | Slower | single |
| matmul__gemv | **1.21x** | Faster | single |
| **Norm** | | | |
| norm__batchnorm | **1.21x** | Faster | single |
| norm__instancenorm | **2.28x** | Faster | single |
| norm__layernorm | **1.59x** | Faster | single |
| norm__rmsnorm | **3.35x** | Faster | single |
| **Other** | | | |
| other__concat | **2.00x** | Faster | single |
| other__deformable | **31.53x** | Faster | single |
| other__embedding | **3.08x** | Faster | single |
| other__gatemlp | **4.78x** | Faster | single |
| other__nms | 0.96x | Slower | single |
| other__scatter | 1.78x | single |

---

## Top Performers (Speedup > 5x)

| Rank | Operator | Speedup |
|------|----------|---------|
| 1 | other__deformable | **31.53x** |
| 2 | attention__rope | **14.17x** |
| 3 | attention__merge_attn_states | **13.25x** |
| 4 | attention__gqa | **9.99x** |
| 5 | other__gatemlp | **4.78x** |

---

## Operators Below Baseline (< 1x)

| Operator | Speedup | Note |
|----------|---------|------|
| matmul__bmm | 0.11x | Significant optimization opportunity |
| matmul__dense | 0.20x | Significant optimization opportunity |
| matmul__gemm | 0.29x | Significant optimization opportunity |
| conv__conv2d | 0.42x | Optimization opportunity |
| attention__mha | 0.59x | Optimization opportunity |
| activation__hardshrink | 0.76x | Slight regression |
| other__nms | 0.96x | Near baseline |

---

## How to Re-run Performance Tests

```bash
# From project root, with GPU available:
cd /data/yuqiuchu/Project/ict-agent
python3 task/task_results/uniopbench/a100_uniopbench_v1/runs/20260308_222632/run_perf_benchmark.py

# Extract from existing logs (no GPU needed):
python3 task/task_results/uniopbench/a100_uniopbench_v1/runs/20260308_222632/run_perf_benchmark.py --from-logs
```

For variant-level average speedup, run each operator's `test.py` with `--variants yaml` and ensure `enable_perf=True` is passed to `create_suite()` in the test script.
