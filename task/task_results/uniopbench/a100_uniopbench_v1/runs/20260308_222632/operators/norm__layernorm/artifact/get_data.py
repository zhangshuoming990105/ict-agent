import torch
import ctypes
import os
import sys
from dataclasses import dataclass
from optest.tools.builder import SEED, load_cuda_kernel


@dataclass
class Params:
    """LayerNorm 参数配置"""

    shape: tuple = (2, 4, 32)
    batch_size: int = 2
    seq_length: int = 4
    d_model: int = 32  # feature dimension for normalization


def get_cuda_argtypes():
    return [
        ctypes.c_void_p,  # x (GPU pointer)
        ctypes.c_void_p,  # gamma (GPU pointer)
        ctypes.c_void_p,  # beta (GPU pointer)
        ctypes.c_void_p,  # output (GPU pointer)
        ctypes.c_int,  # batch_size
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
    gamma = torch.ones(
        params.d_model, dtype=torch.float32, device="cuda"
    )  # learnable scale
    beta = torch.zeros(
        params.d_model, dtype=torch.float32, device="cuda"
    )  # learnable bias

    # Create output tensor
    output_cuda = torch.empty_like(x)
    cuda_output_tensors = [output_cuda]

    cuda_all_inputs = [
        x,
        gamma,
        beta,
        output_cuda,
        params.batch_size,
        params.seq_length,
        params.d_model,
    ]

    # For PyTorch, layer_norm only needs input (gamma and beta are default 1 and 0)
    torch_all_inputs = [x]
    return cuda_all_inputs, torch_all_inputs, cuda_output_tensors


def cuda_output_tensor_transform(cuda_output):
    # For LayerNorm operation, no format transformation needed
    # Both CUDA and PyTorch use the same tensor format
    return cuda_output
