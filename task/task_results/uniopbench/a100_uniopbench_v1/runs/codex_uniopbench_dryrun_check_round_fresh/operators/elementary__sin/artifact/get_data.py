import torch
import ctypes
from dataclasses import dataclass
from optest.tools.checker import SEED


@dataclass
class Params:
    shape: tuple = (7, 1, 6, 7)
    total_elements: int = 294


def get_cuda_argtypes():
    return [
        ctypes.c_void_p,  # input
        ctypes.c_void_p,  # output
        ctypes.c_int,  # total elements
    ]


def get_cuda_torch_inputs(params: Params):
    torch.manual_seed(SEED)
    x = torch.randn(params.shape, dtype=torch.float32, device="cuda")
    output = torch.empty_like(x)
    cuda_all_inputs = [x, output, params.total_elements]
    torch_all_inputs = [x]
    return cuda_all_inputs, torch_all_inputs, [output]


def cuda_output_tensor_transform(tensor):
    return tensor
