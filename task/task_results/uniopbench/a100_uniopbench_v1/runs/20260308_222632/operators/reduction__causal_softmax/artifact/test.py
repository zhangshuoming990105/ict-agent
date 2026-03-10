"""Test script for Causal Softmax operator using the optest framework.

Causal Softmax applies a causal mask (lower triangular) before softmax,
ensuring that position i can only see positions <= i.

Supports two modes:
1. Single test mode (default): Tests with default parameters
2. Variant mode (--variants yaml): Tests all variants from cases.yaml
"""

import os
import sys
import argparse
from dataclasses import dataclass

# Add parent directory to path for imports
TESTCASE_DIR = os.path.dirname(__file__)
sys.path.insert(0, TESTCASE_DIR)
# Add project root to path for optest
PROJECT_ROOT = os.path.abspath(os.path.join(TESTCASE_DIR, "../../../"))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

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


@dataclass
class CausalSoftmaxParams:
    """Causal Softmax test parameters.

    Field names match cases.yaml keys for variant mode support.
    The dtype field is used for dynamic dtype resolution in TensorSpec.
    
    Note: Causal Softmax expects 2D input (batch, seq_len).
    For 3D input (batch, num_heads, seq_len), it will be reshaped to 2D.
    """

    shape: tuple = (8, 16)  # (batch, seq_len)
    dtype: torch.dtype = torch.float32
    
    @property
    def batch_size(self) -> int:
        """Get batch size for CUDA kernel.
        
        For 2D input (batch, seq_len): returns batch
        For 3D input (batch, num_heads, seq_len): returns batch * num_heads
        """
        if len(self.shape) == 2:
            return self.shape[0]
        elif len(self.shape) == 3:
            return self.shape[0] * self.shape[1]  # batch * num_heads
        else:
            raise ValueError(f"Unsupported shape dimension: {len(self.shape)}")
    
    @property
    def seq_len(self) -> int:
        """Get sequence length (last dimension)."""
        return self.shape[-1]
    
    @property
    def total_elements(self) -> int:
        """Total number of elements."""
        result = 1
        for dim in self.shape:
            result *= dim
        return result


# Define test case using new framework
testcase = TestCase(
    tensor_specs=[
        TensorSpec("input", "dtype", ("shape",), "input"),
        TensorSpec("output", "dtype", ("shape",), "output"),
    ],
    scalar_specs=[
        ScalarSpec("batch_size", int, lambda p: p.batch_size),
        ScalarSpec("seq_len", int, lambda p: p.seq_len),
    ],
    torch_kernel=torch_kernel,
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Causal Softmax kernel test (new framework)"
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
        # Causal softmax may have slightly higher numerical errors due to masking
        atol=1e-3,
        rtol=1e-3,
    )

    # Create backend
    if args.backend == "cuda":
        backend = CUDABackend()
    else:  # triton
        backend = TritonBackend()

    # ========== Variant mode: run all variants from cases.yaml ==========
    if args.variants == "yaml":
        suite = create_suite(
            params_class=CausalSoftmaxParams,
            testcase=testcase,
            testcase_dir=TESTCASE_DIR,
            backend=backend,
            variant_source="yaml",
        )
        print(f"\n{'=' * 80}")
        print(f"Testing Causal Softmax operator with {backend.name} backend (variant mode)")
        print(f"{'=' * 80}\n")

        result = suite.run()
        print_suite_summary(result, verbose=args.verbose)
        return result.exit_code

    # ========== Single test mode: test with default parameters ==========
    params = CausalSoftmaxParams()

    print(f"\n{'=' * 80}")
    print(f"Testing Causal Softmax operator with {backend.name} backend (single test mode)")
    print(f"Shape: {params.shape}, Elements: {params.total_elements:,}")
    print(f"Dtype: {params.dtype}")
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
