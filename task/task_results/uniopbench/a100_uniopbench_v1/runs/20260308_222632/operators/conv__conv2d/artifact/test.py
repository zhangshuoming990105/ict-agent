"""Test script for Conv2D operator using the new optest framework.

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
class Conv2DParams:
    """Conv2D test parameters.

    Field names match cases.yaml for variant mode support.
    """

    batch: int = 32
    in_height: int = 8
    in_width: int = 8
    in_channels: int = 128
    out_channels: int = 64
    kernel_height: int = 2
    kernel_width: int = 2
    stride: int = 3
    padding: int = 0
    format: str = "nchw"

    # Computed properties for output dimensions
    @property
    def output_height(self) -> int:
        return (
            self.in_height + 2 * self.padding - self.kernel_height
        ) // self.stride + 1

    @property
    def output_width(self) -> int:
        return (self.in_width + 2 * self.padding - self.kernel_width) // self.stride + 1


# Define test case using new framework
testcase = TestCase(
    tensor_specs=[
        TensorSpec(
            "input",
            torch.float32,
            ("batch", "in_channels", "in_height", "in_width"),
            "input",
        ),
        TensorSpec(
            "kernel",
            torch.float32,
            ("out_channels", "in_channels", "kernel_height", "kernel_width"),
            "input",
        ),
        TensorSpec(
            "output",
            torch.float32,
            ("batch", "out_channels", "output_height", "output_width"),
            "output",
        ),
    ],
    scalar_specs=[
        ScalarSpec("batch", int, lambda p: p.batch),
        ScalarSpec("in_height", int, lambda p: p.in_height),
        ScalarSpec("in_width", int, lambda p: p.in_width),
        ScalarSpec("in_channels", int, lambda p: p.in_channels),
        ScalarSpec("out_channels", int, lambda p: p.out_channels),
        ScalarSpec("kernel_height", int, lambda p: p.kernel_height),
        ScalarSpec("kernel_width", int, lambda p: p.kernel_width),
        ScalarSpec("stride", int, lambda p: p.stride),
        ScalarSpec("padding", int, lambda p: p.padding),
    ],
    torch_kernel=torch_kernel,
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Conv2D kernel test (new framework)")
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
    # Conv2D involves significant floating point accumulation (128 channels * 2x2 kernel)
    # so we use slightly looser tolerances
    config = TestConfig(
        enable_perf=not args.no_perf,
        compile_only=args.compile_only,
        atol=5e-2,  # Absolute tolerance: 0.05
        rtol=5e-2,  # Relative tolerance: 5%
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
            params_class=Conv2DParams,
            testcase=testcase,
            testcase_dir=TESTCASE_DIR,
            backend=backend,
            variant_source="yaml",
            config=config,  # Pass config to suite
        )
        print(f"\n{'=' * 80}")
        print(f"Testing Conv2D operator with {backend.name} backend (variant mode)")
        print(f"{'=' * 80}\n")

        result = suite.run()
        print_suite_summary(result, verbose=args.verbose)
        return result.exit_code

    # Single test mode: test with default parameters
    params = Conv2DParams()

    print(f"\n{'=' * 80}")
    print(f"Testing Conv2D operator with {backend.name} backend (single test mode)")
    print(
        f"Input: ({params.batch}, {params.in_channels}, {params.in_height}, {params.in_width})"
    )
    print(
        f"Kernel: ({params.out_channels}, {params.in_channels}, {params.kernel_height}, {params.kernel_width})"
    )
    print(
        f"Output: ({params.batch}, {params.out_channels}, {params.output_height}, {params.output_width})"
    )
    print(f"Stride: {params.stride}")
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
