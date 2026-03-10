"""Test script for Add operator using the new optest framework.

This is the new unified test entry point that replaces check_cuda.py and
check_triton.py.

Supports two modes:
1. Single test mode (default): Tests with default parameters
2. Variant mode (--variants yaml): Tests all variants from cases.yaml
"""

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
class AddParams:
    shape: tuple = (2048, 2048)
    dtype: torch.dtype = torch.float16
    op_levels: int = 2  # 接收来自 yaml 的值

    @property
    def final_op_type(self) -> int:
        if self.dtype == torch.float32:
            # FP32 有两个版本
            if self.op_levels == 0: return 0  # elementwise_add_f32
            else: return 1                    # elementwise_add_f32x4 
        else:
            # FP16 有四个版本
            if self.op_levels == 0: return 2   # elementwise_add_f16
            elif self.op_levels == 1: return 3 # elementwise_add_f16x2
            elif self.op_levels == 2: return 4  # elementwise_add_f16x8 
            else: return 5  # elementwise_add_f16x8_pack 
    @property
    def total_elements(self) -> int:
        num = 1
        for s in self.shape: num *= s
        return num

testcase = TestCase(
    tensor_specs=[
        TensorSpec("A", "dtype", ("shape",), "input"),
        TensorSpec("B", "dtype", ("shape",), "input"),
        TensorSpec("output", "dtype", ("shape",), "output"),
    ],
    scalar_specs=[
        ScalarSpec("size", int, lambda p: p.total_elements),
        ScalarSpec("op_type", int, lambda p: p.final_op_type),
    ],
    torch_kernel=torch_kernel,
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Add kernel test (new framework)")
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

    # Create test configuration
    config = TestConfig(
        enable_perf=True, # not args.no_perf
        compile_only=args.compile_only,
        verbose=True,  # Suppress verbose output for batch runs
        perf_warmup=100,
        perf_iterations=1000,
        atol=1e-2,
        rtol=1e-2,
    )

    # Create backend
    if args.backend == "cuda":
        backend = CUDABackend()
    else:  # triton
        backend = TritonBackend()

    # Variant mode: run all variants from cases.yaml
    if args.variants == "yaml":
        suite = create_suite(
            params_class=AddParams,
            testcase=testcase,
            testcase_dir=TESTCASE_DIR,
            backend=backend,
            variant_source="yaml",
        )
        print(f"\n{'=' * 80}")
        print(f"Testing Add operator with {backend.name} backend (variant mode)")
        print(f"{'=' * 80}\n")

        result = suite.run()
        print_suite_summary(result, verbose=True) # verbose=args.verbose
        return result.exit_code

    # Single test mode: test with default parameters
    params = AddParams()

    print(f"\n{'=' * 80}")
    print(f"Testing Add operator with {backend.name} backend (single test mode)")
    print(f"Shape: {params.shape}, Elements: {params.total_elements:,}")
    print(f"{'=' * 80}\n")

    result = run_test(
        testcase_dir=TESTCASE_DIR,
        testcase=testcase,
        backend=backend,
        params=params,
        config=config,
    )

    # Print results
    if not args.compile_only:
        print_result(result)

    return 0 if result.passed or result.skipped else 1


if __name__ == "__main__":
    sys.exit(main())
