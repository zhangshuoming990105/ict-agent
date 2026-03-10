import torch
import ctypes
import os
import sys
from dataclasses import dataclass
from torch.nn import functional as F
from optest.tools.builder import SEED, load_cuda_kernel


@dataclass
class Params:
    """DepthwiseConv 参数配置"""

    input_size: int = 128
    output_size: int = 126  # 128 - 3 + 1 = 126
    kernel_size: int = 3
    channels: int = 128


def get_cuda_argtypes():
    return [
        ctypes.c_void_p,  # input (GPU pointer)
        ctypes.c_void_p,  # kernel (GPU pointer)
        ctypes.c_void_p,  # output (GPU pointer)
        ctypes.c_int,  # input_height
        ctypes.c_int,  # kernel_size
        ctypes.c_int,  # input_channels
    ]


def get_cuda_torch_inputs(params: Params):
    """当需要cuda和torch比较时候使用"""
    torch.manual_seed(SEED)
    # DepthwiseConv: HWC format
    # Input: (input_size, input_size, channels) - HWC format
    # Kernel: (kernel_size, kernel_size, channels) - one filter per channel

    # Create data directly on specified device (HWC format)
    input_tensor = torch.randn(
        params.input_size,
        params.input_size,
        params.channels,
        dtype=torch.float32,
        device="cuda",
    ).normal_(mean=0.0, std=0.5)
    kernel_tensor = torch.randn(
        params.kernel_size,
        params.kernel_size,
        params.channels,
        dtype=torch.float32,
        device="cuda",
    ).normal_(mean=0.0, std=0.5)

    # Create output tensor (HWC format)
    output_cuda_hwc = torch.empty(
        params.output_size,
        params.output_size,
        params.channels,
        dtype=torch.float32,
        device="cuda",
    )
    cuda_output_tensors = [output_cuda_hwc]

    cuda_all_inputs = [
        input_tensor,
        kernel_tensor,
        output_cuda_hwc,
        params.input_size,
        params.kernel_size,
        params.channels,
    ]

    # For PyTorch, need to convert HWC to NCHW format
    # Input: HWC -> NCHW (add batch dimension and permute)
    input_nchw = (
        input_tensor.permute(2, 0, 1).unsqueeze(0).contiguous()
    )  # HWC -> NCHW (1, C, H, W)
    # Kernel: HWC -> format needed by PyTorch depthwise conv (channels, 1, kernel_height, kernel_width)
    kernel_nchw = (
        kernel_tensor.permute(2, 0, 1).unsqueeze(1).contiguous()
    )  # HWC -> (C, 1, kH, kW)
    torch_all_inputs = [input_nchw, kernel_nchw]
    return cuda_all_inputs, torch_all_inputs, cuda_output_tensors


def cuda_output_tensor_transform(cuda_output):
    # For DepthwiseConv, convert CUDA output from HWC to NCHW format to match PyTorch output
    # CUDA output: HWC (height, width, channels)
    # PyTorch output: NCHW (batch, channels, height, width)
    # Convert HWC -> NCHW and add batch dimension
    return cuda_output.permute(2, 0, 1).unsqueeze(0).contiguous()  # HWC -> NCHW
