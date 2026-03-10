import torch
import ctypes
from dataclasses import dataclass
from optest.tools.builder import SEED


@dataclass
class Params:
    """SIGMOID operation parameters"""

    shape: tuple = (5, 128)
    total_elements: int = 640  # 5 * 128


def get_cuda_argtypes():
    return [
        ctypes.c_void_p,  # input (GPU pointer)
        ctypes.c_void_p,  # output (GPU pointer)
        ctypes.c_int,  # total_elements
    ]


def get_cuda_torch_inputs(params: Params):
    torch.manual_seed(SEED)
    input_tensor = torch.randn(
        params.shape, dtype=torch.float32, device="cuda"
    ).normal_(mean=0.0, std=0.5)
    output_tensor = torch.empty_like(input_tensor)

    cuda_output_tensors = [output_tensor]

    cuda_all_inputs = [input_tensor, output_tensor, params.total_elements]
    torch_all_inputs = [input_tensor]
    return cuda_all_inputs, torch_all_inputs, cuda_output_tensors


def cuda_output_tensor_transform(cuda_output):
    return cuda_output  # No transformation needed for SIGMOID
