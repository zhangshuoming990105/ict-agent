import torch
import ctypes
from dataclasses import dataclass
from optest.tools.builder import SEED
from optest.tools.layout import convert_nhwc_to_nchw


@dataclass
class Params:
    """AvgPool2D operation parameters"""

    batch_size: int = 4
    height: int = 56
    width: int = 56
    channels: int = 128
    kernel_size: int = 5
    stride: int = 2

    # Calculated output dimensions
    output_height: int = 26  # (56 - 5) // 2 + 1 = 26
    output_width: int = 26  # (56 - 5) // 2 + 1 = 26


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
    torch.manual_seed(SEED)
    # Generate data in NHWC format for CUDA kernel
    input_tensor = (
        torch.randn(
            (params.batch_size, params.channels, params.height, params.width),
            dtype=torch.float32,
            device="cuda",
        )
        .normal_(mean=0.0, std=0.5)
        .to(memory_format=torch.channels_last)
    )

    output_tensor = torch.empty(
        (params.batch_size, params.channels, params.output_height, params.output_width),
        dtype=torch.float32,
        device="cuda",
    ).to(memory_format=torch.channels_last)

    cuda_output_tensors = [output_tensor]

    cuda_all_inputs = [
        input_tensor,
        output_tensor,
        params.batch_size,
        params.channels,
        params.height,
        params.kernel_size,
        params.stride,
    ]
    torch_all_inputs = [input_tensor, params.kernel_size, params.stride]
    return cuda_all_inputs, torch_all_inputs, cuda_output_tensors


def cuda_output_tensor_transform(cuda_output):
    return cuda_output
