"""Test script for Gather operator using the new optest framework.

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
class GatherParams:
    """Gather test parameters."""

    data_shape: tuple = (50, 128, 4)
    num_indices: int = 4
    output_shape: tuple = None
    
    def __post_init__(self):
        # Compute output shape based on data_shape and num_indices
        if self.output_shape is None:
            self.output_shape = (self.data_shape[0], self.data_shape[1], self.num_indices)


# Define test case using new framework
testcase = TestCase(
    tensor_specs=[
        TensorSpec("params", torch.float32, ("data_shape",), "input"),
        TensorSpec("indices", torch.int64, ("num_indices",), "input"),
        TensorSpec(
            "output",
            torch.float32,
            ("output_shape",),
            "output",
        ),
    ],
    scalar_specs=[
        ScalarSpec("dim0", int, lambda p: p.data_shape[0]),
        ScalarSpec("dim1", int, lambda p: p.data_shape[1]),
        ScalarSpec("dim2", int, lambda p: p.data_shape[2]),
        ScalarSpec("num_indices", int, lambda p: p.num_indices),
    ],
    torch_kernel=torch_kernel,
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Gather kernel test (new framework)")
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
    params = GatherParams()

    # Run test
    print(f"\n{'=' * 80}")
    print(f"Testing Gather operator with {backend.name} backend")
    print(f"Data shape: {params.data_shape}, Indices: {params.num_indices}")
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
