import os
import sys
import argparse
from get_data import get_cuda_torch_inputs, Params, cuda_output_tensor_transform
from torch_.ref import torch_kernel
from triton_.kernel import triton_kernel
from optest.tools.checker import compare_results

TESTCASE_ROOT_DIR = os.path.dirname(__file__)
sys.path.insert(0, TESTCASE_ROOT_DIR)


def parse_args():
    parser = argparse.ArgumentParser(
        description="batchnorm_16_3_32_32 Triton kernel test"
    )
    parser.add_argument(
        "--no-perf", action="store_true", help="Placeholder flag for compatibility"
    )
    return parser.parse_args()


def main():
    params = Params()
    _, torch_inputs, _ = get_cuda_torch_inputs(params)
    torch_output = cuda_output_tensor_transform(torch_kernel(*torch_inputs))
    triton_output = cuda_output_tensor_transform(triton_kernel(*torch_inputs))
    compare_results(torch_output, triton_output, test_type=["PyTorch", "Triton"])


if __name__ == "__main__":
    parse_args()
    main()
