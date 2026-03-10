import torch
import ctypes
from dataclasses import dataclass
from optest.tools.builder import SEED


@dataclass
class Params:
    """Conv1D operation parameters"""

    input_size: int = 7
    output_size: int = 5
    kernel_size: int = 3  # derived: input_size - output_size + 1 = 7 - 5 + 1 = 3


def get_cuda_argtypes():
    return [
        ctypes.c_void_p,  # input (GPU pointer)
        ctypes.c_void_p,  # kernel (GPU pointer)
        ctypes.c_void_p,  # output (GPU pointer)
        ctypes.c_int,  # input_size
        ctypes.c_int,  # output_size
    ]


def get_cuda_torch_inputs(params: Params):
    torch.manual_seed(SEED)
    input_tensor = torch.randn(
        (params.input_size,), dtype=torch.float32, device="cuda"
    ).normal_(mean=0.0, std=0.5)
    kernel_tensor = torch.randn(
        (params.kernel_size,), dtype=torch.float32, device="cuda"
    ).normal_(mean=0.0, std=0.5)
    output_tensor = torch.empty(
        (params.output_size,), dtype=torch.float32, device="cuda"
    )

    cuda_output_tensors = [output_tensor]

    cuda_all_inputs = [
        input_tensor,
        kernel_tensor,
        output_tensor,
        params.input_size,
        params.output_size,
    ]
    torch_all_inputs = [input_tensor, kernel_tensor]
    return cuda_all_inputs, torch_all_inputs, cuda_output_tensors


def cuda_output_tensor_transform(cuda_output):
    return cuda_output  # No transformation needed for Conv1D
