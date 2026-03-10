import torch
import ctypes
from dataclasses import dataclass
from optest.tools.checker import SEED


@dataclass
class Params:
    input_shape: tuple = (16, 32, 32)
    output_shape: tuple = (32, 32)
    rows: int = 16
    inner: int = 1024


def get_cuda_argtypes():
    return [
        ctypes.c_void_p,  # input
        ctypes.c_void_p,  # output
        ctypes.c_int,  # rows
        ctypes.c_int,  # inner
    ]


def get_cuda_torch_inputs(params: Params):
    torch.manual_seed(SEED)
    x = torch.randn(params.input_shape, dtype=torch.float32, device="cuda")
    output = torch.empty(params.output_shape, dtype=torch.float32, device="cuda")
    cuda_all_inputs = [x, output, params.rows, params.inner]
    torch_all_inputs = [x]
    return cuda_all_inputs, torch_all_inputs, [output]


def cuda_output_tensor_transform(tensor):
    return tensor
