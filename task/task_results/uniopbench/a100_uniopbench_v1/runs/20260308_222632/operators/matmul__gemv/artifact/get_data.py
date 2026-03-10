import torch
import ctypes
import os
import sys
from dataclasses import dataclass
from optest.tools.builder import SEED, load_cuda_kernel


@dataclass
class Params:
    """GEMV (General Matrix Vector Multiplication) 参数配置"""

    m: int = 32  # matrix rows
    n: int = 512  # matrix cols / vector length
    # A shape: (m, n)
    # x shape: (n,)
    # y shape: (m,)


def get_cuda_argtypes():
    return [
        ctypes.c_void_p,  # A (GPU pointer)
        ctypes.c_void_p,  # x (GPU pointer)
        ctypes.c_void_p,  # y (GPU pointer)
        ctypes.c_int,  # m
        ctypes.c_int,  # n
    ]


def get_cuda_torch_inputs(params: Params):
    """当需要cuda和torch比较时候使用"""
    torch.manual_seed(SEED)
    # GEMV operation: A(m,n) @ x(n) = y(m)
    A = torch.randn(params.m, params.n, dtype=torch.float32, device="cuda").normal_(
        mean=0.0, std=0.5
    )
    x = torch.randn(params.n, dtype=torch.float32, device="cuda").normal_(
        mean=0.0, std=0.5
    )

    # Create output vector
    y = torch.empty(params.m, dtype=torch.float32, device="cuda")
    cuda_output_tensors = [y]

    cuda_all_inputs = [A, x, y, params.m, params.n]

    # For PyTorch, inputs are already in correct format
    torch_all_inputs = [A, x]
    return cuda_all_inputs, torch_all_inputs, cuda_output_tensors


def cuda_output_tensor_transform(cuda_output):
    # For GEMV operation, no format transformation needed
    # Both CUDA and PyTorch use the same tensor format
    return cuda_output
