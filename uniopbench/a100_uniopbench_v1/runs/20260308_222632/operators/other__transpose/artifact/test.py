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
class TransposeParams:
    # 约定形状: (M, N)
    shape: tuple = (1024, 1024)
    dtype: torch.dtype = torch.float32
    op_levels: int = 13 # 0-13 对应 kernel.cu 中的 switch case

    @property
    def in_shape(self) -> tuple:
        return (int(self.shape[0]), int(self.shape[1]))

    @property
    def out_shape(self) -> tuple:
        return (int(self.shape[1]), int(self.shape[0]))

    @property
    def M(self) -> int:
        return int(self.shape[0])

    @property
    def N(self) -> int:
        return int(self.shape[1])
    
    @property
    def op_levels_val(self) -> int:
        return int(self.op_levels)


testcase = TestCase(
    tensor_specs=[
        TensorSpec("input", "dtype", (lambda p: p.in_shape), "input"),
        TensorSpec("output", "dtype", (lambda p: p.out_shape), "output"),
    ],
    scalar_specs=[
        ScalarSpec("M", int, lambda p: p.M),
        ScalarSpec("N", int, lambda p: p.N),
        ScalarSpec("op_levels", int, lambda p: p.op_levels_val),
    ],
    torch_kernel=torch_kernel,
)

def manual_generate_data(self, params):
    device = torch.device("cuda")

    val_M = params.M
    val_N = params.N
    val_op = params.op_levels_val

    # 生成数据: M x N
    input_tensor = torch.randn(
        val_M, val_N, 
        dtype=torch.float32, device=device
    )
    
    # 输出: N x M
    output_tensor = torch.zeros(
        val_N, val_M, 
        dtype=torch.float32, device=device
    )

    all_inputs = [
        input_tensor,
        output_tensor,
        val_M,
        val_N,
        val_op
    ]

    torch_inputs = [input_tensor]

    return {
        "all_inputs": all_inputs,
        "torch_inputs": torch_inputs,
        "outputs": [output_tensor]   
    }

testcase.generate_data = manual_generate_data.__get__(testcase)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Transpose kernel test (new framework)")
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
        verbose=True,
        perf_warmup=100,
        perf_iterations=1000,
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
            params_class=TransposeParams,
            testcase=testcase,
            testcase_dir=TESTCASE_DIR,
            backend=backend,
            variant_source="yaml",
        )
        print(f"\n{'=' * 80}")
        print(f"Testing Transpose operator with {backend.name} backend (variant mode)")
        print(f"{'=' * 80}\n")

        result = suite.run()
        print_suite_summary(result, verbose=True)
        return result.exit_code

    # Single test mode: test with default parameters
    params = TransposeParams()

    print(f"\n{'=' * 80}")
    print(f"Testing Transpose operator with {backend.name} backend (single test mode)")
    print(f"Shape: {params.shape}")
    print(f"Dtype: {params.dtype}, Op Level: {params.op_levels} -> Kernel OpType: {params.op_levels}")
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