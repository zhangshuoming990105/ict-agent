import torch
import ctypes
import os
import sys
from dataclasses import dataclass
from torch.nn import functional as F
from optest.tools.builder import SEED, load_cuda_kernel


@dataclass
class Params:
    """Conv2DNCHW 参数配置"""

    batch_size: int = 32
    input_height: int = 8
    input_width: int = 8
    input_channels: int = 128
    output_channels: int = 64
    kernel_height: int = 2
    kernel_width: int = 2
    stride: int = 3
    padding: int = 0
    output_height: int = 3
    output_width: int = 3


def get_cuda_argtypes():
    return [
        ctypes.c_void_p,  # input (GPU pointer)
        ctypes.c_void_p,  # kernel (GPU pointer)
        ctypes.c_void_p,  # output (GPU pointer)
        ctypes.c_int,  # batch_size
        ctypes.c_int,  # input_height
        ctypes.c_int,  # input_channels
        ctypes.c_int,  # output_channels
        ctypes.c_int,  # kernel_height
        ctypes.c_int,  # stride
    ]


def get_cuda_torch_inputs(params: Params):
    """当需要cuda和torch比较时候使用"""
    torch.manual_seed(SEED)
    # Conv2DNCHW: conv2dnchw_16_64_8_8_128_64_2_2_2_0 (NCHW format)
    # Input: (N, C, H, W) = (batch_size, input_channels, input_height, input_width)
    # Kernel: (O, C, kH, kW) = (output_channels, input_channels, kernel_height, kernel_width)

    # Create data directly on specified device (NCHW format)
    input_tensor = torch.randn(
        params.batch_size,
        params.input_channels,
        params.input_height,
        params.input_width,
        dtype=torch.float32,
        device="cuda",
    ).normal_(mean=0.0, std=0.5)
    kernel_tensor = torch.randn(
        params.output_channels,
        params.input_channels,
        params.kernel_height,
        params.kernel_width,
        dtype=torch.float32,
        device="cuda",
    ).normal_(mean=0.0, std=0.5)

    # Create output tensor (NCHW format for CUDA)
    output_cuda_nchw = torch.empty(
        params.batch_size,
        params.output_channels,
        params.output_height,
        params.output_width,
        dtype=torch.float32,
        device="cuda",
    )
    cuda_output_tensors = [output_cuda_nchw]

    cuda_all_inputs = [
        input_tensor,
        kernel_tensor,
        output_cuda_nchw,
        params.batch_size,
        params.input_height,
        params.input_channels,
        params.output_channels,
        params.kernel_height,
        params.stride,
    ]

    # For PyTorch, input and kernel are already in NCHW format, so no conversion needed
    torch_all_inputs = [input_tensor, kernel_tensor]
    return cuda_all_inputs, torch_all_inputs, cuda_output_tensors


def cuda_output_tensor_transform(cuda_output):
    # For NCHW format, no transformation needed as both CUDA and PyTorch use NCHW
    return cuda_output
