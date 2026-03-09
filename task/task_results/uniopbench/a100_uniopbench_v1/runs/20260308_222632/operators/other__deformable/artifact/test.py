"""Test script for Deformable Attention operator using the new optest framework.

This is the new unified test entry point that replaces check_cuda.py and
check_triton.py.

Note: Deformable Attention is a complex operator that uses MultiScaleDeformableAttention
module. This test provides unified support for both CUDA and Triton backends.
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
from get_data import get_cuda_torch_inputs, Params, get_cuda_argtypes

# Add parent directory to path for imports
TESTCASE_DIR = os.path.dirname(__file__)
sys.path.insert(0, TESTCASE_DIR)


try:
    from get_data import cuda_output_tensor_transform
except ImportError:
    cuda_output_tensor_transform = lambda x: x


@dataclass
class DeformableParams:
    """Deformable Attention test parameters."""

    batch_size: int = 1  # n
    num_queries: int = 200  # lq
    num_heads: int = 8  # m
    embed_dim: int = 512  # d
    num_levels: int = 4  # l
    num_points: int = 4  # k


class DeformableTestCase(TestCase):
    """Custom TestCase for deformable attention with custom data generation."""
    
    def generate_data(self, params, device="cuda"):
        """Generate test data using the custom get_data.py function."""
        # Convert DeformableParams to Params from get_data.py
        data_params = Params(
            batch_size=params.batch_size,
            num_queries=params.num_queries,
            num_heads=params.num_heads,
            embed_dim=params.embed_dim,
            num_levels=params.num_levels,
            num_points=params.num_points,
        )
        
        cuda_inputs, torch_inputs, cuda_outputs = get_cuda_torch_inputs(data_params)
        
        return {
            "all_inputs": cuda_inputs,
            "torch_inputs": torch_inputs,
            "outputs": cuda_outputs,
        }


# Define test case using new framework with custom data generation
testcase = DeformableTestCase(
    tensor_specs=[],  # Not used with custom generate_data
    scalar_specs=[],
    torch_kernel=torch_kernel,
    argtypes=get_cuda_argtypes(),
    output_transform=cuda_output_tensor_transform,
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Deformable kernel test (new framework)"
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
    params = DeformableParams()

    # Run test
    print(f"\n{'=' * 80}")
    print(f"Testing Deformable Attention operator with {backend.name} backend")
    print(
        f"Batch: {params.batch_size}, Queries: {params.num_queries}, Heads: {params.num_heads}"
    )
    print(
        f"Embed dim: {params.embed_dim}, Levels: {params.num_levels}, Points: {params.num_points}"
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
