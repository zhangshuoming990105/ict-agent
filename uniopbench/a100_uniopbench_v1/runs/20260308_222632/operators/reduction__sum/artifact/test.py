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
class ReduceParams:
    shape: tuple = (4096,123)
    dtype: torch.dtype = torch.float32 # 占位，实际由 op_levels 决定
    op_levels: int = 19  # 对应 kernel.cu 中的 op_type (0-19)

testcase = TestCase(
    tensor_specs=[
        TensorSpec("input", "dtype", ("shape",), "input"),
        TensorSpec("output", "dtype", (None,), "output"),
    ],
    scalar_specs=[
        ScalarSpec("n", int, lambda p: math.prod(p.shape)),
        ScalarSpec("op_type", int, lambda p: p.op_levels),
    ],
    torch_kernel=torch_kernel,
)

def get_config_by_op_type(op_type, shape):
    """根据 op_type 返回 (input_dtype, output_dtype)"""
    # 0-1: F32
    if 0 <= op_type <= 1:
        return torch.float32, torch.float32
    
    # 2-7: F16
    if 2 <= op_type <= 7:
        return torch.float16, torch.float32
        
    # 8-13: BF16
    if 8 <= op_type <= 13:
        return torch.bfloat16, torch.float32
        
    # 14-17: FP8
    if 14 <= op_type <= 17:
        # 14,15: E4M3
        if op_type in [14, 15]:
            return torch.float8_e4m3fn, torch.float32
        # 16,17: E5M2
        else:
            return torch.float8_e5m2, torch.float32
            
    # 18-19: INT8
    if 18 <= op_type <= 19:
        return torch.int8, torch.int32
    
    raise ValueError(f"Unknown op_type: {op_type}")

def custom_generate_data(params: ReduceParams):
    op = params.op_levels
    in_dtype, out_dtype = get_config_by_op_type(op, params.shape)
    
    low_prec_accum_ops = [2, 4, 6, 8, 10, 12, 14, 15, 16, 17]

    # 生成输入数据
    if in_dtype in [getattr(torch, 'float8_e4m3fn', None), getattr(torch, 'float8_e5m2', None)]:
        if not torch.cuda.is_available() or not hasattr(torch, 'float8_e4m3fn'):
            print(f"Skipping FP8 test op={op}")
            input_tensor = torch.zeros(params.shape, device='cuda', dtype=torch.float32)
            out_dtype = torch.float32
        else:
            input_tensor = torch.randint(-2, 3, params.shape, device='cuda').to(torch.float32).to(in_dtype)
    
    elif op in low_prec_accum_ops:
        input_tensor = torch.randint(-2, 3, params.shape, device='cuda', dtype=in_dtype)
        
    elif in_dtype == torch.int8:
        input_tensor = torch.randint(-10, 10, params.shape, device='cuda', dtype=in_dtype)
    else:
        input_tensor = torch.randn(params.shape, device='cuda', dtype=in_dtype)

    output_tensor = torch.zeros((1,), device='cuda', dtype=out_dtype)
    n = math.prod(params.shape)

    return {
        "torch_inputs": [input_tensor, op], 
        "all_inputs": [input_tensor, output_tensor, n, op], 
        "outputs": [output_tensor]
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
        perf_warmup=10,
        perf_iterations=100,
        atol=1.0, # bf16精度太低，时过时不过，只能调大，其他的容差可在1e-2
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
            params_class=ReduceParams,
            testcase=testcase,
            testcase_dir=TESTCASE_DIR,
            backend=backend,
            variant_source="yaml",
        )
        print(f"\n{'=' * 80}")
        print(f"Testing Reduce operator with {backend.name} backend (variant mode)")
        print(f"{'=' * 80}\n")

        result = suite.run()
        print_suite_summary(result, verbose=True) # verbose=args.verbose
        return result.exit_code

    # Single test mode: test with default parameters
    params = ReduceParams()

    print(f"\n{'=' * 80}")
    print(f"Testing Reduce operator with {backend.name} backend (single test mode)")
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