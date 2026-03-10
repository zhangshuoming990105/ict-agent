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
class EmbeddingParams:
    # 约定形状: (SeqLen, MaxV, EmbSize)
    shape: tuple = (2048, 1024, 512)
    dtype: torch.dtype = torch.float16 # float32 float16
    op_levels: int = 2

    @property
    def idx_shape(self) -> tuple:
        return (int(self.shape[0]),)

    @property
    def w_shape(self) -> tuple:
        return (int(self.shape[1]), int(self.shape[2]))

    @property
    def out_shape(self) -> tuple:
        return (int(self.shape[0]), int(self.shape[2]))

    @property
    def N(self) -> int:
        return int(self.shape[0])

    @property
    def emb_size(self) -> int:
        return int(self.shape[2])

    @property
    def op_type(self) -> int:
        base = 0 if self.dtype == torch.float32 else 3
        level = max(0, min(self.op_levels, 2))
        return int(base + level)
    
    @property
    def max_v(self) -> int:
        return int(self.shape[1])


# 定义测试用例
testcase = TestCase(
    tensor_specs=[
        TensorSpec("indices", torch.int32, (lambda p: p.idx_shape), "input"), # 索引用int类型的特殊处理
        TensorSpec("weight", "dtype", (lambda p: p.w_shape), "input"),
        TensorSpec("output", "dtype", (lambda p: p.out_shape), "output"),

    ],
    scalar_specs=[
        ScalarSpec("N", int, lambda p: p.N),
        ScalarSpec("emb_size", int, lambda p: p.emb_size),
        ScalarSpec("op_type", int, lambda p: p.op_type),
    ],
    torch_kernel=torch_kernel,
)

def manual_generate_data(self, params):
    device = torch.device("cuda")

    val_N = params.N
    val_emb = params.emb_size
    val_max_v = params.max_v
    val_op = params.op_type

    indices = torch.randint(
        low=0, high=val_max_v, size=(val_N,), 
        dtype=torch.int32, device=device
    )
    
    weight = torch.randn(
        val_max_v, val_emb, 
        dtype=params.dtype, device=device
    )
    
    output = torch.zeros(
        val_N, val_emb, 
        dtype=params.dtype, device=device
    )

    all_inputs = [
        indices,
        weight,
        output,
        val_N,
        val_emb,
        val_op
    ]

    torch_inputs = [indices, weight]

    return {
        "all_inputs": all_inputs,
        "torch_inputs": torch_inputs,
        "outputs": [output]   
    }

testcase.generate_data = manual_generate_data.__get__(testcase)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Embedding kernel test (new framework)")
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
            params_class=EmbeddingParams,
            testcase=testcase,
            testcase_dir=TESTCASE_DIR,
            backend=backend,
            variant_source="yaml",
        )
        print(f"\n{'=' * 80}")
        print(f"Testing Embedding operator with {backend.name} backend (variant mode)")
        print(f"{'=' * 80}\n")

        result = suite.run()
        print_suite_summary(result, verbose=True)
        return result.exit_code

    # Single test mode: test with default parameters
    params = EmbeddingParams()

    print(f"\n{'=' * 80}")
    print(f"Testing Embedding operator with {backend.name} backend (single test mode)")
    print(f"Shape: {params.shape}")
    print(f"Dtype: {params.dtype}, Op Level: {params.op_levels} -> Kernel OpType: {params.op_type}")
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