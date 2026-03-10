import torch
import ctypes
import os
import sys
from dataclasses import dataclass
from optest.tools.builder import SEED, load_cuda_kernel


@dataclass
class Params:
    """Multi-Head Attention 参数配置"""

    batch_size: int = 64
    seq_len: int = 2048  # N_CTX
    num_heads: int = 12  # H
    head_dim: int = 512  # D_HEAD
    scale_factor: float = 22.6  # sqrt(D_HEAD) = sqrt(512) = 22.6


def get_cuda_argtypes():
    return [
        ctypes.c_void_p,  # q (GPU pointer)
        ctypes.c_void_p,  # k (GPU pointer)
        ctypes.c_void_p,  # v (GPU pointer)
        ctypes.c_void_p,  # output (GPU pointer)
        ctypes.c_int,  # batch_size
        ctypes.c_int,  # seq_len
        ctypes.c_int,  # num_heads
        ctypes.c_int,  # head_dim
    ]


def get_cuda_torch_inputs(params: Params):
    """当需要cuda和torch比较时候使用"""
    torch.manual_seed(SEED)
    # Create Q, K, V tensors with shape (B, N_CTX, H, D_HEAD)
    shape = (params.batch_size, params.seq_len, params.num_heads, params.head_dim)

    q = torch.randn(shape, dtype=torch.float32, device="cuda").normal_(
        mean=0.0, std=0.5
    )
    k = torch.randn(shape, dtype=torch.float32, device="cuda").normal_(
        mean=0.0, std=0.5
    )
    v = torch.randn(shape, dtype=torch.float32, device="cuda").normal_(
        mean=0.0, std=0.5
    )

    # Create output tensor
    output_cuda = torch.empty_like(q)
    cuda_output_tensors = [output_cuda]

    cuda_all_inputs = [
        q,
        k,
        v,
        output_cuda,
        params.batch_size,
        params.seq_len,
        params.num_heads,
        params.head_dim,
    ]

    # For PyTorch, inputs are already in correct format
    torch_all_inputs = [q, k, v]
    return cuda_all_inputs, torch_all_inputs, cuda_output_tensors


def cuda_output_tensor_transform(cuda_output):
    # For MHA operation, no format transformation needed
    # Both CUDA and PyTorch use the same tensor format
    return cuda_output
