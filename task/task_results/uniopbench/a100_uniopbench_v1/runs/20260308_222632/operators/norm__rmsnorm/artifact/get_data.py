import torch
import ctypes
import os
import sys
from dataclasses import dataclass
from optest.tools.builder import SEED, load_cuda_kernel


@dataclass
class Params:
    """RMSNorm 参数配置"""

    seq_length: int = 4096
    d_model: int = 4096  # feature dimension for normalization
    shape: tuple = (4096, 4096)
    eps: float = 1e-6


def get_cuda_argtypes():
    return [
        ctypes.c_void_p,  # x (GPU pointer)
        ctypes.c_void_p,  # output (GPU pointer)
        ctypes.c_int,  # seq_length
        ctypes.c_int,  # d_model
    ]


def get_cuda_torch_inputs(params: Params):
    """当需要cuda和torch比较时候使用"""
    torch.manual_seed(SEED)
    # Create data directly on specified device
    x = torch.randn(params.shape, dtype=torch.float32, device="cuda").normal_(
        mean=0.0, std=0.5
    )

    # Create output tensor
    output_cuda = torch.empty_like(x)
    cuda_output_tensors = [output_cuda]

    cuda_all_inputs = [x, output_cuda, params.seq_length, params.d_model]

    # For PyTorch, inputs are already in correct format
    torch_all_inputs = [x]
    return cuda_all_inputs, torch_all_inputs, cuda_output_tensors


def cuda_output_tensor_transform(cuda_output):
    # For RMSNorm operation, no format transformation needed
    # Both CUDA and PyTorch use the same tensor format
    return cuda_output
