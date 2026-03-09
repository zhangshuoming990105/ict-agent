"""Test script for Square operator using the optest framework."""

import os
import sys
import argparse
from dataclasses import dataclass, field
from functools import reduce
from operator import mul

import torch
from optest import (
    TestCase,
    TensorSpec,
    ScalarSpec,
    TestConfig,
    CUDABackend,
    run_test,
    print_result,
    create_suite,
    print_suite_summary,
)

from torch_.ref import torch_kernel


TESTCASE_DIR = os.path.dirname(__file__)
sys.path.insert(0, TESTCASE_DIR)


def _num_elements(shape: tuple) -> int:
    """Compute total number of elements for a given shape."""
    if len(shape) == 0:
        return 1
    return int(reduce(mul, shape, 1))


@dataclass
class SquareParams:
    """Square operator test parameters.

    Field names must match keys in cases.yaml (shape, dtype).
    """

    shape: tuple = (256,)
    dtype: torch.dtype = torch.float32
    total_elements: int = field(default=256)

    def __post_init__(self):
        # Keep total_elements consistent with shape for both default
        # single-test mode and YAML-driven variant mode.
        self.total_elements = _num_elements(self.shape)


testcase = TestCase(
    tensor_specs=[
        TensorSpec("input", "dtype", ("shape",), "input"),
        TensorSpec("output", "dtype", ("shape",), "output"),
    ],
    scalar_specs=[
        ScalarSpec("size", int, lambda p: p.total_elements),
    ],
    torch_kernel=torch_kernel,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Square operator test")
    parser.add_argument(
        "--compile-only",
        action="store_true",
        help="Only compile the kernel without running tests.",
    )
    parser.add_argument(
        "--no-perf",
        action="store_true",
        help="Skip performance benchmarking.",
    )
    parser.add_argument(
        "--backend",
        choices=["cuda"],
        default="cuda",
        help="Backend to test (currently only cuda is supported).",
    )
    parser.add_argument(
        "--variants",
        choices=["yaml", "none"],
        default="none",
        help='Variant mode: "yaml" to load from cases.yaml, "none" for single test.',
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print detailed variant results in variant mode.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    config = TestConfig(
        enable_perf=not args.no_perf,
        compile_only=args.compile_only,
        verbose=False,
        atol=1e-2,
        rtol=1e-2,
    )

    backend = CUDABackend()

    # Variant mode: load all shapes/dtypes from cases.yaml
    if args.variants == "yaml":
        suite = create_suite(
            params_class=SquareParams,
            testcase=testcase,
            testcase_dir=TESTCASE_DIR,
            backend=backend,
            variant_source="yaml",
        )

        print(f"\n{'=' * 80}")
        print(f"Testing Square with {backend.name} backend (variant mode)")
        print(f"{'=' * 80}\n")

        result = suite.run()
        print_suite_summary(result, verbose=args.verbose)
        return result.exit_code

    # Single-test mode
    params = SquareParams()

    print(f"\n{'=' * 80}")
    print(f"Testing Square with {backend.name} backend (single test mode)")
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

    if not args.compile_only:
        print_result(result)

    return 0 if result.passed or result.skipped else 1


if __name__ == "__main__":
    sys.exit(main())

