"""Test script for BMM operator using the new optest framework.

This is the new unified test entry point that replaces check_cuda.py and
check_triton.py.
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
class BMMParams:
    """BMM test parameters."""

    batch_size: int = 4
    dim1: int = 512  # A.shape = (batch_size, dim1, dim2)
    dim2: int = 512  # B.shape = (batch_size, dim2, dim3)
    dim3: int = 512  # C.shape = (batch_size, dim1, dim3)


# Define test case using new framework
testcase = TestCase(
    tensor_specs=[
        TensorSpec("A", torch.float16, ("batch_size", "dim1", "dim2"), "input"),
        TensorSpec("B", torch.float16, ("batch_size", "dim2", "dim3"), "input"),
        TensorSpec("C", torch.float32, ("batch_size", "dim1", "dim3"), "output"),
    ],
    scalar_specs=[
        ScalarSpec("b", int, lambda p: p.batch_size),
        ScalarSpec("m", int, lambda p: p.dim1),
        ScalarSpec("k", int, lambda p: p.dim2),
        ScalarSpec("n", int, lambda p: p.dim3),
    ],
    torch_kernel=torch_kernel,
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="BMM kernel test (new framework)")
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
    config = TestConfig(
        enable_perf=not args.no_perf,
        compile_only=args.compile_only,
    )

    # Create backend
    if args.backend == "cuda":
        backend = CUDABackend()
    else:  # triton
        backend = TritonBackend()

    # Create test parameters
    params = BMMParams()

    # Run test
    print(f"\n{'=' * 80}")
    print(f"Testing BMM operator with {backend.name} backend")
    print(
        f"A: ({params.batch_size}, {params.dim1}, {params.dim2}), "
        f"B: ({params.batch_size}, {params.dim2}, {params.dim3}), "
        f"C: ({params.batch_size}, {params.dim1}, {params.dim3})"
    )
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
