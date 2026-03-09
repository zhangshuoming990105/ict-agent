import os
import sys
import argparse
import math
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
class DotProductParams:
    shape: tuple = (1024,1024)
    dtype: torch.dtype = torch.float32 
    op_levels: int = 1 

testcase = TestCase(
    tensor_specs=[
        TensorSpec("a", "dtype", ("shape",), "a"),
        TensorSpec("b", "dtype", ("shape",), "b"),
        TensorSpec("y", "dtype", (None,), "y"),
    ],
    scalar_specs=[
        ScalarSpec("n", int, lambda p: math.prod(p.shape)),
        ScalarSpec("op_type", int, lambda p: p.op_levels),
    ],
    torch_kernel=torch_kernel,
)

def get_dtype_by_op(op_type):
    # 0:f32, 1:f32x4
    if op_type in [0, 1]:
        return torch.float32
    # 2:f16, 3:f16x2, 4:f16x8
    if op_type in [2, 3, 4]:
        return torch.float16
    return torch.float32

def custom_generate_data(params: DotProductParams):
    op = params.op_levels
    dtype = get_dtype_by_op(op)
    
    # [精度控制]
    if dtype == torch.float16:
        input_a = torch.randint(-2, 3, params.shape, device='cuda').to(dtype)
        input_b = torch.randint(-2, 3, params.shape, device='cuda').to(dtype)
    else:
        input_a = torch.randn(params.shape, device='cuda', dtype=dtype)
        input_b = torch.randn(params.shape, device='cuda', dtype=dtype)

    output_y = torch.zeros((1,), device='cuda', dtype=torch.float32)
    
    n = math.prod(params.shape)

    return {
        "torch_inputs": [input_a, input_b, op],
        "all_inputs": [input_a, input_b, output_y, n, op],
        "outputs": [output_y]
    }

testcase.generate_data = custom_generate_data

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
        atol=1e-2,
        rtol=1e-2,
    )

    # Create backend
    if args.backend == "cuda":
        backend = CUDABackend()
    else:  # triton
        backend = TritonBackend()

    # Variant mode: run all variants from cases.yaml
    if args.variants == "yaml":
        suite = create_suite(
            params_class=DotProductParams,
            testcase=testcase,
            testcase_dir=TESTCASE_DIR,
            backend=backend,
            variant_source="yaml",
        )
        print(f"\n{'=' * 80}")
        print(f"Testing DotProduct operator with {backend.name} backend (variant mode)")
        print(f"{'=' * 80}\n")

        result = suite.run()
        print_suite_summary(result, verbose=True) # verbose=args.verbose
        return result.exit_code

    # Single test mode: test with default parameters
    params = DotProductParams()

    print(f"\n{'=' * 80}")
    print(f"Testing DotProduct operator with {backend.name} backend (single test mode)")
    print(f"Shape: {params.shape}")
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