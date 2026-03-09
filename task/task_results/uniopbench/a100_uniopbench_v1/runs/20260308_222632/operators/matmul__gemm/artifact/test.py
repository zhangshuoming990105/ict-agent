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
class GemmParams:
    shape: tuple = (1024, 1024, 1024)
    dtype: torch.dtype = torch.float32
    op_levels: int = 3   
    
    def __post_init__(self):
        # Kernel 0~3 是 SGEMM (Float32)
        if self.op_levels <= 3:
            self.dtype = torch.float32
        # Kernel 4~7 是 HGEMM (Float16)
        else:
            self.dtype = torch.float16

    @property
    def M(self): return self.shape[0]
    @property
    def N(self): return self.shape[1]
    @property
    def K(self): return self.shape[2]

    @property
    def final_op_type(self) -> int:
        return self.op_levels

    @property
    def shape_a(self):
        return (self.M, self.K)
    
    @property
    def shape_b(self):
        return (self.K, self.N)
    
    @property
    def shape_c(self):
        return (self.M, self.N)

testcase = TestCase(
    tensor_specs=[
        TensorSpec("a", "dtype", ("shape_a",), "input"),
        TensorSpec("b", "dtype", ("shape_b",), "input"),
        TensorSpec("c", "dtype", ("shape_c",), "output"),
    ],
    scalar_specs=[
        ScalarSpec("M", int, lambda p: p.M),
        ScalarSpec("N", int, lambda p: p.N),
        ScalarSpec("K", int, lambda p: p.K),
        ScalarSpec("op_type", int, lambda p: p.final_op_type),
    ],
    torch_kernel=torch_kernel,
)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Gemm kernel test (new framework)")
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
    args = parse_args()

    config = TestConfig(
        enable_perf=not args.no_perf,
        compile_only=args.compile_only,
        verbose=True,
        perf_warmup=20,       
        perf_iterations=100,  
        atol=1e-1,           
        rtol=1e-1,
    )

    if args.backend == "cuda":
        backend = CUDABackend()
    else:
        backend = TritonBackend()

    if args.variants == "yaml":
        suite = create_suite(
            params_class=GemmParams,
            testcase=testcase,
            testcase_dir=TESTCASE_DIR,
            backend=backend,
            variant_source="yaml",
            config=config 
        )
        print(f"\n{'=' * 80}")
        print(f"Testing GEMM operator (Direct Mapping Mode 0-11)")
        print(f"{'=' * 80}\n")
        
        result = suite.run()
        print_suite_summary(result, verbose=True)
        return result.exit_code

    # Single test mode
    params = GemmParams()

    print(f"\n{'=' * 80}")
    print(f"Testing GEMM Single Mode")
    print(f"Kernel ID: {params.op_levels} -> Auto Dtype: {params.dtype}")
    print(f"Shape: M={params.M}, N={params.N}, K={params.K}")
    print(f"{'=' * 80}\n")

    result = run_test(TESTCASE_DIR, testcase, backend, params, config)
    if not args.compile_only:
        print_result(result)

    return 0 if result.passed or result.skipped else 1

if __name__ == "__main__":
    sys.exit(main())