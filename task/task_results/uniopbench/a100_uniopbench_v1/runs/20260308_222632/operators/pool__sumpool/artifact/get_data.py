import torch
import ctypes
import os
import sys
from dataclasses import dataclass
from optest.tools.builder import SEED, load_cuda_kernel
from optest.tools.layout import convert_nhwc_to_nchw


@dataclass
class Params:
    """SumPool 参数配置"""

    batch_size: int = 16
    input_height: int = 64
    input_width: int = 64
    channels: int = 64
    kernel_height: int = 5
    kernel_width: int = 5
    stride: int = 1
    padding: int = 1  # parsed from folder name
    output_height: int = 60  # (64 - 5) // 1 + 1 = 60
    output_width: int = 60  # (64 - 5) // 1 + 1 = 60


def get_cuda_argtypes():
    return [
        ctypes.c_void_p,  # input (GPU pointer)
        ctypes.c_void_p,  # output (GPU pointer)
        ctypes.c_int,  # batch_size
        ctypes.c_int,  # channels
        ctypes.c_int,  # input_H
        ctypes.c_int,  # kernel_size
        ctypes.c_int,  # stride
    ]


def get_cuda_torch_inputs(params: Params):
    """当需要cuda和torch比较时候使用"""
    torch.manual_seed(SEED)
    # Generate data in NHWC format directly for CUDA kernel: (batch_size, height, width, channels)
    shape_nhwc = (
        params.batch_size,
        params.input_height,
        params.input_width,
        params.channels,
    )  # NHWC format

    # Create data directly on specified device in NHWC format
    x_nhwc = torch.randn(shape_nhwc, dtype=torch.float32, device="cuda").normal_(
        mean=0.0, std=0.5
    )

    # Calculate output dimensions
    output_shape_nhwc = (
        params.batch_size,
        params.output_height,
        params.output_width,
        params.channels,
    )  # NHWC format

    # Create output tensor in NHWC format
    output_cuda_nhwc = torch.empty(
        output_shape_nhwc, dtype=torch.float32, device="cuda"
    )
    cuda_output_tensors = [output_cuda_nhwc]

    cuda_all_inputs = [
        x_nhwc,
        output_cuda_nhwc,
        params.batch_size,
        params.channels,
        params.input_height,
        params.kernel_height,
        params.stride,
    ]

    # For PyTorch, convert from NHWC to NCHW format
    x_nchw = convert_nhwc_to_nchw(x_nhwc)
    torch_all_inputs = [x_nchw, params.kernel_height, params.stride]
    return cuda_all_inputs, torch_all_inputs, cuda_output_tensors


def cuda_output_tensor_transform(cuda_output):
    # For SumPool, convert CUDA output from NHWC to NCHW format to match PyTorch output
    # CUDA output: NHWC (batch, height, width, channels)
    # PyTorch output: NCHW (batch, channels, height, width)
    return convert_nhwc_to_nchw(cuda_output)
