"""Test script for Mean operator using the new optest framework.

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
class MeanParams:
    """Mean test parameters."""

    input_shape: tuple = (1, 64)
    output_shape: tuple = (64,)
    rows: int = 1
    inner: int = 64


# Define test case using new framework
testcase = TestCase(
    tensor_specs=[
        TensorSpec("input", torch.float32, ("input_shape",), "input"),
        TensorSpec("output", torch.float32, ("output_shape",), "output"),
    ],
    scalar_specs=[
        ScalarSpec("rows", int, lambda p: p.rows),
        ScalarSpec("inner", int, lambda p: p.inner),
    ],
    torch_kernel=torch_kernel,
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Mean kernel test (new framework)")
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
    params = MeanParams()

    # Run test
    print(f"\n{'=' * 80}")
    print(f"Testing Mean operator with {backend.name} backend")
    print(f"Input shape: {params.input_shape}, Output shape: {params.output_shape}")
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
