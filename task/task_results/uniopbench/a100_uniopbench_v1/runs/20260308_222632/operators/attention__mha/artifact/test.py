"""Test script for MHA operator using the new optest framework.

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
class MHAParams:
    """MHA test parameters."""

    batch_size: int = 64
    seq_len: int = 2048  # N_CTX
    num_heads: int = 12  # H
    head_dim: int = 512  # D_HEAD
    scale_factor: float = 22.6  # sqrt(D_HEAD) = sqrt(512) = 22.6


# Define test case using new framework
testcase = TestCase(
    tensor_specs=[
        TensorSpec(
            "q",
            torch.float32,
            ("batch_size", "seq_len", "num_heads", "head_dim"),
            "input",
        ),
        TensorSpec(
            "k",
            torch.float32,
            ("batch_size", "seq_len", "num_heads", "head_dim"),
            "input",
        ),
        TensorSpec(
            "v",
            torch.float32,
            ("batch_size", "seq_len", "num_heads", "head_dim"),
            "input",
        ),
        TensorSpec(
            "output",
            torch.float32,
            ("batch_size", "seq_len", "num_heads", "head_dim"),
            "output",
        ),
    ],
    scalar_specs=[
        ScalarSpec("batch_size", int, lambda p: p.batch_size),
        ScalarSpec("seq_len", int, lambda p: p.seq_len),
        ScalarSpec("num_heads", int, lambda p: p.num_heads),
        ScalarSpec("head_dim", int, lambda p: p.head_dim),
    ],
    torch_kernel=torch_kernel,
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="MHA kernel test (new framework)")
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
    # Note: MHA has large inputs (64, 2048, 12, 512), so use reduced benchmark iterations
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
    params = MHAParams()

    # Run test
    print(f"\n{'=' * 80}")
    print(f"Testing MHA operator with {backend.name} backend")
    print(
        f"Shape: ({params.batch_size}, {params.seq_len}, {params.num_heads}, {params.head_dim})"
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
