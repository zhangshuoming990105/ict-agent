"""Test script for GQA operator using the new optest framework.

This is the new unified test entry point that replaces check_cuda.py and
check_triton.py.

Note: GQA was missing check_triton.py in the old framework. This new test.py
provides unified support for both CUDA and Triton backends.
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
)

from torch_.ref import torch_kernel

# Add parent directory to path for imports
TESTCASE_DIR = os.path.dirname(__file__)
sys.path.insert(0, TESTCASE_DIR)


@dataclass
class GQAParams:
    """GQA test parameters: 16 Q heads share 2 KV heads (group_size=8)."""

    batch: int = 1
    num_q_heads: int = 16
    num_kv_heads: int = 2
    M: int = 16  # SEQ_Q
    K_dim: int = 16  # HEAD_DIM
    N: int = 512  # SEQ_KV


# Define test case using new framework
testcase = TestCase(
    tensor_specs=[
        TensorSpec("Q", torch.float16, ("batch", "num_q_heads", "M", "K_dim"), "input"),
        TensorSpec(
            "K", torch.float16, ("batch", "num_kv_heads", "N", "K_dim"), "input"
        ),
        TensorSpec(
            "V", torch.float16, ("batch", "num_kv_heads", "N", "K_dim"), "input"
        ),
        TensorSpec(
            "O", torch.float16, ("batch", "num_q_heads", "M", "K_dim"), "output"
        ),
    ],
    scalar_specs=[
        ScalarSpec("batch", int, lambda p: p.batch),
        ScalarSpec("num_q_heads", int, lambda p: p.num_q_heads),
        ScalarSpec("num_kv_heads", int, lambda p: p.num_kv_heads),
        ScalarSpec("M", int, lambda p: p.M),
        ScalarSpec("K_dim", int, lambda p: p.K_dim),
        ScalarSpec("N", int, lambda p: p.N),
    ],
    torch_kernel=torch_kernel,
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="GQA kernel test (new framework)")
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
    return parser.parse_args()


def main():
    """Main test entry point."""
    args = parse_args()

    # Create test configuration
    # Note: GQA has moderate input sizes, use reduced benchmark iterations for faster testing
    config = TestConfig(
        enable_perf=not args.no_perf,
        compile_only=args.compile_only,
        perf_warmup=10,  # Reduced from default 100
        perf_iterations=50,  # Reduced from default 1000
    )

    # Create backend
    if args.backend == "cuda":
        backend = CUDABackend()
    else:  # triton
        backend = TritonBackend()

    # Create test parameters
    params = GQAParams()

    # Run test
    print(f"\n{'=' * 80}")
    print(f"Testing GQA operator with {backend.name} backend")
    print(f"Q heads: {params.num_q_heads}, KV heads: {params.num_kv_heads}")
    print(f"Q: ({params.batch * params.num_q_heads}, {params.M}, {params.K_dim})")
    print(f"K/V: ({params.batch * params.num_kv_heads}, {params.N}, {params.K_dim})")
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
