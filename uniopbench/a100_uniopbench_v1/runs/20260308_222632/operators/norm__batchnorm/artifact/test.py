"""Test script for BatchNorm operator using the new optest framework.

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
class BatchNormParams:
    """BatchNorm test parameters.

    Field names match cases.yaml keys for variant mode support.
    The dtype field is used for dynamic dtype resolution in TensorSpec.

    All fields are directly defined in cases.yaml (named parameter format).
    """

    shape: tuple = (16, 3, 32, 32)
    batch_size: int = 16
    num_channels: int = 3
    spatial_size: int = 1024
    eps: float = 1e-5
    dtype: torch.dtype = torch.float32  # Dynamic dtype (from cases.yaml or default)


def init_positive_variance(shape, dtype, device):
    """Initialize variance tensor with positive values (variance must be > 0)."""
    return torch.rand(*shape, dtype=dtype, device=device) + 0.5  # Range [0.5, 1.5]


# Define test case using new framework
# Note: dtype is specified as "dtype" string reference, which will be resolved from params
testcase = TestCase(
    tensor_specs=[
        TensorSpec("input", "dtype", ("shape",), "input"),  # Dynamic dtype from params
        TensorSpec("output", "dtype", ("shape",), "output"),
        TensorSpec("running_mean", "dtype", ("num_channels",), "input"),
        TensorSpec(
            "running_var",
            "dtype",
            ("num_channels",),
            "input",
            init_fn=init_positive_variance,
        ),
        TensorSpec("gamma", "dtype", ("num_channels",), "input"),
        TensorSpec("beta", "dtype", ("num_channels",), "input"),
    ],
    scalar_specs=[
        ScalarSpec("batch_size", int, lambda p: p.batch_size),
        ScalarSpec("num_channels", int, lambda p: p.num_channels),
        ScalarSpec("spatial_size", int, lambda p: p.spatial_size),
    ],
    torch_kernel=torch_kernel,
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="BatchNorm kernel test (new framework)"
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
        enable_perf=not args.no_perf,
        compile_only=args.compile_only,
        verbose=False,  # Suppress verbose output for batch runs
    )

    # Create backend
    if args.backend == "cuda":
        backend = CUDABackend()
    else:  # triton
        backend = TritonBackend()

    # Variant mode: run all variants from cases.yaml
    if args.variants == "yaml":
        suite = create_suite(
            params_class=BatchNormParams,
            testcase=testcase,
            testcase_dir=TESTCASE_DIR,
            backend=backend,
            variant_source="yaml",
        )
        print(f"\n{'=' * 80}")
        print(f"Testing BatchNorm operator with {backend.name} backend (variant mode)")
        print(f"{'=' * 80}\n")

        result = suite.run()
        print_suite_summary(result, verbose=args.verbose)
        return result.exit_code

    # Single test mode: test with default parameters
    params = BatchNormParams()

    print(f"\n{'=' * 80}")
    print(f"Testing BatchNorm operator with {backend.name} backend (single test mode)")
    print(f"Shape: {params.shape}, Channels: {params.num_channels}")
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
