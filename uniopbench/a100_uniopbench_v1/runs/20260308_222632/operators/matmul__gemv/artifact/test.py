import os
import sys
import argparse
from dataclasses import dataclass

import torch
from optest import (
    TestCase,
    TensorSpec,
    ScalarSpec,
    TestConfig,
    CUDABackend,
    TritonBackend,
    run_test,
    print_result,
    create_suite,
    print_suite_summary,
)

from torch_.ref import torch_kernel

# Add parent directory to path for imports
TESTCASE_DIR = os.path.dirname(__file__)
sys.path.insert(0, TESTCASE_DIR)

@dataclass
class GemvParams:
    shape: tuple = (1024, 128)  # (M, K)
    dtype: torch.dtype = torch.float32
    op_levels: int = 2
    
    @property
    def M(self):
        return self.shape[0]

    @property
    def K(self):
        return self.shape[1]

    # 使k自动适配
    @property
    def final_op_type(self) -> int:
        base = 0 if self.dtype == torch.float32 else 3
        
        # Case 1: K=16，必须使用 k16 kernel (index 0)
        if self.K == 16:
            return base + 0
            
        # Case 2: K 是 128 的倍数，且显式请求了向量化优化 (op_level=2)
        if self.K % 128 == 0 and self.op_levels == 2:
            return base + 2
            
        # Case 3: 默认兜底 (K32, K64, 或未指定优化的 K128)
        # k32 kernel (index 1) 是最通用的，支持所有 32 倍数的 K
        return base + 1

    @property
    def shape_a(self):
        return (self.M, self.K)
    
    @property
    def shape_x(self):
        return (self.K, 1)
    
    @property
    def shape_y(self):
        return (self.M, 1)

testcase = TestCase(
    tensor_specs=[
        TensorSpec("a", "dtype", ("shape_a",), "input"),
        TensorSpec("x", "dtype", ("shape_x",), "input"),
        TensorSpec("y", "dtype", ("shape_y",), "output"),
    ],
    scalar_specs=[
        ScalarSpec("M", int, lambda p: p.M),
        ScalarSpec("K", int, lambda p: p.K),
        ScalarSpec("op_type", int, lambda p: p.final_op_type),
    ],
    torch_kernel=torch_kernel,
)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Gemv kernel test (new framework)")
    parser.add_argument(
        "--compile-only",
        action="store_true",
        help="Only compile the kernel without running tests",
    )
    parser.add_argument(
        "--no-perf", action="store_true", help="Disable performance testing"
    )
    parser.add_argument(
        "--backend",
        choices=["cuda", "triton"],
        default="cuda",
        help="Backend to test (default: cuda)",
    )
    parser.add_argument(
        "--variants",
        choices=["yaml", "none"],
        default="none",
        help='Test variants: "yaml" to load from cases.yaml, "none" for single test',
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print individual variant results in suite mode",
    )
    return parser.parse_args()

def main():
    """Main test entry point."""
    args = parse_args()

    config = TestConfig(
        enable_perf=not args.no_perf,
        compile_only=args.compile_only,
        verbose=True,
        perf_warmup=100,
        perf_iterations=1000,
        atol=5e-2, # f16精度不够，需要调大
        rtol=5e-2,
    )

    if args.backend == "cuda":
        backend = CUDABackend()
    else:
        backend = TritonBackend()

    if args.variants == "yaml":
        suite = create_suite(
            params_class=GemvParams,
            testcase=testcase,
            testcase_dir=TESTCASE_DIR,
            backend=backend,
            variant_source="yaml",
            config=config
        )
        print(f"\n{'=' * 80}")
        print(f"Testing Gemv operator with {backend.name} backend (variant mode)")
        print(f"{'=' * 80}\n")
        result = suite.run()
        print_suite_summary(result, verbose=True)
        return result.exit_code

    # Single test mode
    params = GemvParams()
    print(f"\n{'=' * 80}")
    print(f"Testing Gemv operator with {backend.name} backend (single test mode)")
    print(f"Shape: {params.shape}, Dtype: {params.dtype}")
    print(f"Op Level: {params.op_levels} -> Final OpType: {params.final_op_type}")
    print(f"{'=' * 80}\n")

    result = run_test(TESTCASE_DIR, testcase, backend, params, config)
    if not args.compile_only:
        print_result(result)

    return 0 if result.passed or result.skipped else 1

if __name__ == "__main__":
    sys.exit(main())