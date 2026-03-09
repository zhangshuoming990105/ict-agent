import torch
import ctypes
from dataclasses import dataclass
from optest.tools.builder import SEED
from optest.tools.layout import convert_nhwc_to_nchw


@dataclass
class Params:
    """MaxPool2D operation parameters"""

    batch_size: int = 4
    height: int = 8
    width: int = 8
    channels: int = 64
    kernel_size: int = 5
    stride: int = 3

    # Calculated output dimensions
    output_height: int = 2  # (8 - 5) // 3 + 1 = 2
    output_width: int = 2  # (8 - 5) // 3 + 1 = 2


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
    input_tensor_nchw = (
        torch.randn(
            (params.batch_size, params.channels, params.height, params.width),
            dtype=torch.float32,
            device="cuda",
        )
        .normal_(mean=0.0, std=0.5)
        .to(memory_format=torch.channels_last)
    )

    output_tensor_nchw = torch.empty(
        (params.batch_size, params.channels, params.output_height, params.output_width),
        dtype=torch.float32,
        device="cuda",
    ).to(memory_format=torch.channels_last)

    cuda_output_tensors = [output_tensor_nchw]

    cuda_all_inputs = [
        input_tensor_nchw,
        output_tensor_nchw,
        params.batch_size,
        params.channels,
        params.height,  # input_H
        params.kernel_size,
        params.stride,
    ]
    torch_all_inputs = [input_tensor_nchw, params.kernel_size, params.stride]
    return cuda_all_inputs, torch_all_inputs, cuda_output_tensors


def cuda_output_tensor_transform(cuda_output):
    return cuda_output


# def get_cuda_torch_inputs(params: Params):
#     torch.manual_seed(SEED)
#     # Generate data in NHWC format for CUDA kernel
#     input_tensor_nhwc = torch.randn(
#         (params.batch_size, params.height, params.width, params.channels),
#         dtype=torch.float32, device="cuda"
#     ).normal_(mean=0.0, std=0.5)

#     output_tensor_nhwc = torch.empty(
#         (params.batch_size, params.output_height, params.output_width, params.channels),
#         dtype=torch.float32, device="cuda"
#     )

#     cuda_output_tensors = [output_tensor_nhwc]

#     # Convert to NCHW for PyTorch
#     input_tensor_nchw = convert_nhwc_to_nchw(input_tensor_nhwc)

#     cuda_all_inputs = [
#         input_tensor_nhwc,
#         output_tensor_nhwc,
#         params.batch_size,
#         params.channels,
#         params.height,  # input_H
#         params.kernel_size,
#         params.stride
#     ]
#     torch_all_inputs = [input_tensor_nchw, params.kernel_size, params.stride]
#     return cuda_all_inputs, torch_all_inputs, cuda_output_tensors

# def cuda_output_tensor_transform(cuda_output_nhwc):
#     # Convert CUDA output from NHWC to NCHW for comparison with PyTorch
#     return convert_nhwc_to_nchw(cuda_output_nhwc)
