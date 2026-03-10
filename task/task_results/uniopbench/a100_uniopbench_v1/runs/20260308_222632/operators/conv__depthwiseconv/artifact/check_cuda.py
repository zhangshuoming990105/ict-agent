import os
import sys
import argparse
from get_data import get_cuda_torch_inputs, Params, get_cuda_argtypes  # type: ignore

try:
    from get_data import cuda_output_tensor_transform  # type: ignore
except ImportError:
    cuda_output_tensor_transform = lambda x: x

from torch_.ref import torch_kernel  # type: ignore
from optest.tools.checker import check_cuda_vs_torch

TESTCASE_ROOT_DIR = os.path.dirname(__file__)

sys.path.insert(0, TESTCASE_ROOT_DIR)


def parse_args():
    parser = argparse.ArgumentParser(description="CUDA kernel test")
    parser.add_argument(
        "--compile-only",
        action="store_true",
        help="Only compile the CUDA kernel without running tests (default: False)",
    )
    parser.add_argument(
        "--no-perf",
        action="store_true",
        help="Disable performance testing (default: False, performance testing enabled)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    enable_perf = not args.no_perf  # Default is True, disable with --no-perf
    compile_only = args.compile_only  # Default is False
    params = Params()

    check_cuda_vs_torch(
        testcase_root_dir=TESTCASE_ROOT_DIR,
        get_cuda_torch_inputs=get_cuda_torch_inputs,
        params=params,
        torch_kernel=torch_kernel,
        get_cuda_argtypes=get_cuda_argtypes,
        output_tensor_transform=cuda_output_tensor_transform,
        enable_perf=enable_perf,
        compile_only=compile_only,
    )
