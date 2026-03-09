"""Test script for DepthwiseConv operator using the new optest framework.

This is the new unified test entry point that replaces check_cuda.py and
check_triton.py.

Note: DepthwiseConv uses HWC format for CUDA kernel and requires output transform.
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

from torch_.ref import torch_kernel as _torch_kernel_ref

# Add parent directory to path for imports
TESTCASE_DIR = os.path.dirname(__file__)
sys.path.insert(0, TESTCASE_DIR)


@dataclass
class DepthwiseConvParams:
    """DepthwiseConv test parameters."""

    input_size: int = 128
    output_size: int = 126  # 128 - 3 + 1 = 126
    kernel_size: int = 3
    channels: int = 128


# Wrapper for torch kernel to handle HWC -> NCHW input transformation
def torch_kernel(input_tensor, kernel_tensor):
    """Wrapper that transforms HWC inputs to NCHW for PyTorch reference."""
    # Input: HWC -> NCHW (add batch dimension and permute)
    input_nchw = input_tensor.permute(2, 0, 1).unsqueeze(0).contiguous()
    # Kernel: HWC -> (C, 1, kH, kW)
    kernel_nchw = kernel_tensor.permute(2, 0, 1).unsqueeze(1).contiguous()
    return _torch_kernel_ref(input_nchw, kernel_nchw)


# Output transform: HWC -> NCHW
def depthwise_output_transform(output):
    """Convert CUDA output from HWC to NCHW format."""
    return output.permute(2, 0, 1).unsqueeze(0).contiguous()


# Define test case using new framework with HWC output transform
testcase = TestCase(
    tensor_specs=[
        TensorSpec(
            "input", torch.float32, ("input_size", "input_size", "channels"), "input"
        ),
        TensorSpec(
            "kernel", torch.float32, ("kernel_size", "kernel_size", "channels"), "input"
        ),
        TensorSpec(
            "output",
            torch.float32,
            ("output_size", "output_size", "channels"),
            "output",
        ),
    ],
    scalar_specs=[
        ScalarSpec("input_height", int, lambda p: p.input_size),
        ScalarSpec("kernel_size", int, lambda p: p.kernel_size),
        ScalarSpec("input_channels", int, lambda p: p.channels),
    ],
    torch_kernel=torch_kernel,
    output_transform=depthwise_output_transform,
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="DepthwiseConv kernel test (new framework)"
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
    params = DepthwiseConvParams()

    # Run test
    print(f"\n{'=' * 80}")
    print(f"Testing DepthwiseConv operator with {backend.name} backend")
    print(f"Input: ({params.input_size}, {params.input_size}, {params.channels}) [HWC]")
    print(f"Kernel: ({params.kernel_size}, {params.kernel_size}, {params.channels})")
    print(f"Output: ({params.output_size}, {params.output_size}, {params.channels})")
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
