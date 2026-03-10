import torch
import ctypes
from dataclasses import dataclass
from optest.tools.checker import SEED


@dataclass
class Params:
    input_shape: tuple = (4, 32, 112, 112)
    output_shape: tuple = (4, 64, 112, 112)
    N: int = 4
    C: int = 32
    H: int = 112
    W: int = 112


def get_cuda_argtypes():
    return [
        ctypes.c_void_p,  # input1
        ctypes.c_void_p,  # input2
        ctypes.c_void_p,  # output
        ctypes.c_int,  # N
        ctypes.c_int,  # C
        ctypes.c_int,  # H
        ctypes.c_int,  # W
    ]


def get_cuda_torch_inputs(params: Params):
    torch.manual_seed(SEED)
    a = torch.randn(params.input_shape, dtype=torch.float32, device="cuda")
    b = torch.randn(params.input_shape, dtype=torch.float32, device="cuda")
    output = torch.empty(params.output_shape, dtype=torch.float32, device="cuda")
    cuda_all_inputs = [a, b, output, params.N, params.C, params.H, params.W]
    torch_all_inputs = [a, b]
    return cuda_all_inputs, torch_all_inputs, [output]


def cuda_output_tensor_transform(tensor):
    return tensor
