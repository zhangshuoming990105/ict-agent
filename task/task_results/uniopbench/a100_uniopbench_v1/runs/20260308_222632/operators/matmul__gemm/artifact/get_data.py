import torch
import ctypes
import os
import sys
from dataclasses import dataclass
from optest.tools.builder import SEED, load_cuda_kernel


@dataclass
class Params:
    """GEMM (General Matrix Multiplication) 参数配置"""

    m: int = 32  # A matrix rows
    k: int = 32  # A matrix cols / B matrix rows
    n: int = 128  # B matrix cols
    # A shape: (m, k)
    # B shape: (k, n)
    # C shape: (m, n)


def get_cuda_argtypes():
    return [
        ctypes.c_void_p,  # A (GPU pointer)
        ctypes.c_void_p,  # B (GPU pointer)
        ctypes.c_void_p,  # C (GPU pointer)
        ctypes.c_int,  # m
        ctypes.c_int,  # k
        ctypes.c_int,  # n
    ]


def get_cuda_torch_inputs(params: Params):
    """当需要cuda和torch比较时候使用"""
    torch.manual_seed(SEED)
    # GEMM operation: A(m,k) @ B(k,n) = C(m,n)
    A = torch.randn(params.m, params.k, dtype=torch.float16, device="cuda").normal_(
        mean=0.0, std=0.5
    )
    B = torch.randn(params.k, params.n, dtype=torch.float16, device="cuda").normal_(
        mean=0.0, std=0.5
    )

    # Create output tensor (float32 for GEMM)
    C = torch.empty(params.m, params.n, dtype=torch.float32, device="cuda")
    cuda_output_tensors = [C]

    cuda_all_inputs = [A, B, C, params.m, params.k, params.n]

    # For PyTorch, inputs are already in correct format
    torch_all_inputs = [A, B]
    return cuda_all_inputs, torch_all_inputs, cuda_output_tensors


def cuda_output_tensor_transform(cuda_output):
    # For GEMM operation, no format transformation needed
    # Both CUDA and PyTorch use the same tensor format
    return cuda_output
