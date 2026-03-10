import torch
import ctypes
from dataclasses import dataclass
from optest.tools.checker import SEED


@dataclass
class Params:
    shape: tuple = (2, 16, 1024)
    total_elements: int = 32768


def get_cuda_argtypes():
    return [
        ctypes.c_void_p,  # a
        ctypes.c_void_p,  # b
        ctypes.c_void_p,  # output
        ctypes.c_int,  # total elements
    ]


def get_cuda_torch_inputs(params: Params):
    torch.manual_seed(SEED)
    a = torch.randn(params.shape, dtype=torch.float32, device="cuda")
    b = torch.randn(params.shape, dtype=torch.float32, device="cuda")
    output = torch.empty_like(a)
    cuda_all_inputs = [a, b, output, params.total_elements]
    torch_all_inputs = [a, b]
    return cuda_all_inputs, torch_all_inputs, [output]


def cuda_output_tensor_transform(tensor):
    return tensor
