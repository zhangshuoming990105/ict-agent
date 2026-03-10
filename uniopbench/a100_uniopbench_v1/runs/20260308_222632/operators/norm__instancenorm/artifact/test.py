"""Test script for InstanceNorm operator using the new optest framework.

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
class InstanceNormParams:
    """InstanceNorm test parameters."""

    shape: tuple = (1, 3, 224, 224)
    batch: int = 1
    channels: int = 3
    spatial: int = 50176


# Define test case using new framework
testcase = TestCase(
    tensor_specs=[
        TensorSpec("x", torch.float32, ("shape",), "input"),
        TensorSpec("output", torch.float32, ("shape",), "output"),
        TensorSpec("gamma", torch.float32, ("channels",), "input"),
        TensorSpec("beta", torch.float32, ("channels",), "input"),
    ],
    scalar_specs=[
        ScalarSpec("batch", int, lambda p: p.batch),
        ScalarSpec("channels", int, lambda p: p.channels),
        ScalarSpec("spatial", int, lambda p: p.spatial),
    ],
    torch_kernel=torch_kernel,
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="InstanceNorm kernel test (new framework)"
    )
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
    params = InstanceNormParams()

    # Run test
    print(f"\n{'=' * 80}")
    print(f"Testing InstanceNorm operator with {backend.name} backend")
    print(f"Shape: {params.shape}, Channels: {params.channels}")
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
