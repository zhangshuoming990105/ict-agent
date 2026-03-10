import os
import sys
import argparse
from get_data import get_cuda_torch_inputs, Params, cuda_output_tensor_transform  # type: ignore
from torch_.ref import torch_kernel  # type: ignore
from triton_.kernel import triton_kernel  # type: ignore
from optest.tools.checker import check_triton_vs_torch

TESTCASE_ROOT_DIR = os.path.dirname(__file__)

sys.path.insert(0, TESTCASE_ROOT_DIR)


def parse_args():
    parser = argparse.ArgumentParser(description="Triton kernel test")
    parser.add_argument(
        "--no-perf",
        action="store_true",
        help="Disable performance testing (default: False, performance testing enabled)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    enable_perf = not args.no_perf  # Default is True, disable with --no-perf
    params = Params()
    check_triton_vs_torch(
        get_cuda_torch_inputs,
        params,
        torch_kernel,
        triton_kernel,
        cuda_output_tensor_transform,
        enable_perf=enable_perf,
    )
