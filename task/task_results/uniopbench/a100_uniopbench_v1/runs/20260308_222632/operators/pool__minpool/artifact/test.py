"""Test script for MinPool operator using the new optest framework.

This is the new unified test entry point that replaces check_cuda.py and
check_triton.py.

Note: MinPool uses NHWC format for CUDA kernel and requires output transform.
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

from torch_.ref import torch_kernel as _torch_kernel_nchw
from optest.tools.layout import convert_nhwc_to_nchw

# Add parent directory to path for imports
TESTCASE_DIR = os.path.dirname(__file__)
sys.path.insert(0, TESTCASE_DIR)


# Wrap torch_kernel to convert NHWC input to NCHW
def torch_kernel(x: torch.Tensor, kernel_size: int, stride: int) -> torch.Tensor:
    """Wrapper that converts NHWC input to NCHW for PyTorch reference."""
    x_nchw = convert_nhwc_to_nchw(x)
    return _torch_kernel_nchw(x_nchw, kernel_size, stride)


@dataclass
class MinPoolParams:
    """MinPool test parameters."""

    batch_size: int = 4
    input_height: int = 35
    input_width: int = 35
    channels: int = 192
    kernel_size: int = 5
    stride: int = 3
    output_height: int = 11  # (35 - 5) // 3 + 1 = 11
    output_width: int = 11  # (35 - 5) // 3 + 1 = 11


# Define test case using new framework with NHWC output transform
testcase = TestCase(
    tensor_specs=[
        TensorSpec(
            "input",
            torch.float32,
            ("batch_size", "input_height", "input_width", "channels"),
            "input",
        ),
        TensorSpec(
            "output",
            torch.float32,
            ("batch_size", "output_height", "output_width", "channels"),
            "output",
        ),
    ],
    scalar_specs=[
        ScalarSpec("kernel_size", int, lambda p: p.kernel_size),
        ScalarSpec("stride", int, lambda p: p.stride),
        ScalarSpec("batch_size", int, lambda p: p.batch_size),
        ScalarSpec("channels", int, lambda p: p.channels),
        ScalarSpec("input_H", int, lambda p: p.input_height),
    ],
    torch_kernel=torch_kernel,
    output_transform=convert_nhwc_to_nchw,
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="MinPool kernel test (new framework)")
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
    params = MinPoolParams()

    # Run test
    print(f"\n{'=' * 80}")
    print(f"Testing MinPool operator with {backend.name} backend")
    print(
        f"Input: ({params.batch_size}, {params.input_height}, {params.input_width}, {params.channels}) [NHWC]"
    )
    print(
        f"Output: ({params.batch_size}, {params.output_height}, {params.output_width}, {params.channels})"
    )
    print(f"Kernel: {params.kernel_size}, Stride: {params.stride}")
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
