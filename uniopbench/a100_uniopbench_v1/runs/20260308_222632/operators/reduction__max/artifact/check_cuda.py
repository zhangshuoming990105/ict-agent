import os
import sys
import argparse
from get_data import (
    get_cuda_torch_inputs,
    Params,
    get_cuda_argtypes,
    cuda_output_tensor_transform,
)
from torch_.ref import torch_kernel
from optest.tools.checker import check_cuda_vs_torch

TESTCASE_ROOT_DIR = os.path.dirname(__file__)
sys.path.insert(0, TESTCASE_ROOT_DIR)


def parse_args():
    parser = argparse.ArgumentParser(description="max_16_32_32 CUDA kernel test")
    parser.add_argument(
        "--compile-only",
        action="store_true",
        help="Only compile the CUDA kernel without running correctness checks",
    )
    parser.add_argument(
        "--no-perf", action="store_true", help="Disable performance benchmarking"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    params = Params()
    check_cuda_vs_torch(
        testcase_root_dir=TESTCASE_ROOT_DIR,
        get_cuda_torch_inputs=get_cuda_torch_inputs,
        params=params,
        torch_kernel=torch_kernel,
        get_cuda_argtypes=get_cuda_argtypes,
        output_tensor_transform=cuda_output_tensor_transform,
        enable_perf=not args.no_perf,
        compile_only=args.compile_only,
    )
