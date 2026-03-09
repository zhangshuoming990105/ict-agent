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

TESTCASE_DIR = os.path.dirname(__file__)
sys.path.insert(0, TESTCASE_DIR)

@dataclass
class MergeAttnParams:
    num_tokens: int = 128
    num_heads: int = 32
    head_size: int = 128
    dtype: torch.dtype = torch.float16 # torch.float16 torch.float32

    @property
    def dtype_code(self) -> int:
        return 0 if self.dtype == torch.float32 else 1

testcase = TestCase(
    tensor_specs=[
        TensorSpec("output", "dtype", ("num_tokens", "num_heads", "head_size"), "output"),
        TensorSpec("output_lse", torch.float32, ("num_heads", "num_tokens"), "output_lse"),
        TensorSpec("prefix_output", "dtype", ("num_tokens", "num_heads", "head_size"), "input"),
        TensorSpec("prefix_lse", torch.float32, ("num_heads", "num_tokens"), "input"),
        TensorSpec("suffix_output", "dtype", ("num_tokens", "num_heads", "head_size"), "input"),
        TensorSpec("suffix_lse", torch.float32, ("num_heads", "num_tokens"), "input"),
    ],
    scalar_specs=[
        ScalarSpec("num_tokens", int, lambda p: p.num_tokens),
        ScalarSpec("num_heads", int, lambda p: p.num_heads),
        ScalarSpec("head_size", int, lambda p: p.head_size),
        ScalarSpec("dtype_code", int, lambda p: p.dtype_code),
    ],
    torch_kernel=torch_kernel,
)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Add kernel test (new framework)")
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
        enable_perf=True, # not args.no_perf
        compile_only=args.compile_only,
        verbose=True,  # Suppress verbose output for batch runs
        perf_warmup=20,
        perf_iterations=100,
        atol=1e-3,
        rtol=1e-3,
    )

    # Create backend
    if args.backend == "cuda":
        backend = CUDABackend()
    else:  # triton
        backend = TritonBackend()

    # Variant mode: run all variants from cases.yaml
    if args.variants == "yaml":
        suite = create_suite(
            params_class=MergeAttnParams,
            testcase=testcase,
            testcase_dir=TESTCASE_DIR,
            backend=backend,
            variant_source="yaml",
        )
        print(f"\n{'=' * 80}")
        print(f"Testing MergeAttn operator with {backend.name} backend (variant mode)")
        print(f"{'=' * 80}\n")

        result = suite.run()
        print_suite_summary(result, verbose=True) # verbose=args.verbose
        return result.exit_code

    # Single test mode: test with default parameters
    params = MergeAttnParams()

    print(f"\n{'=' * 80}")
    print(f"Testing MergeAttn operator with {backend.name} backend (single test mode)")
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