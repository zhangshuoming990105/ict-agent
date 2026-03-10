"""Test script for GateMLP operator using the new optest framework.

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
class GateMLPParams:
    """GateMLP test parameters."""

    batch: int = 32
    k: int = 128
    n: int = 128


# Define test case using new framework
testcase = TestCase(
    tensor_specs=[
        TensorSpec("X", torch.float16, ("batch", "k"), "input"),
        TensorSpec("A", torch.float16, ("k", "n"), "input"),
        TensorSpec("B", torch.float16, ("k", "n"), "input"),
        TensorSpec("output", torch.float32, ("batch", "n"), "output"),
    ],
    scalar_specs=[
        ScalarSpec("batch", int, lambda p: p.batch),
        ScalarSpec("K", int, lambda p: p.k),
        ScalarSpec("N", int, lambda p: p.n),
    ],
    torch_kernel=torch_kernel,
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="GateMLP kernel test (new framework)")
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
    params = GateMLPParams()

    # Run test
    print(f"\n{'=' * 80}")
    print(f"Testing GateMLP operator with {backend.name} backend")
    print(f"X: ({params.batch}, {params.k}), A/B: ({params.k}, {params.n})")
    print(f"Output: ({params.batch}, {params.n})")
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
