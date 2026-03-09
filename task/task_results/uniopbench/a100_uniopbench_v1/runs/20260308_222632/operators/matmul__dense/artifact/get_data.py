import torch
import ctypes
from dataclasses import dataclass
from optest.tools.checker import SEED


@dataclass
class Params:
    m: int = 64
    k: int = 768
    n: int = 768


def get_cuda_argtypes():
    return [
        ctypes.c_void_p,  # A
        ctypes.c_void_p,  # B
        ctypes.c_void_p,  # bias
        ctypes.c_void_p,  # output
        ctypes.c_int,  # M
        ctypes.c_int,  # K
        ctypes.c_int,  # N
    ]


def get_cuda_torch_inputs(params: Params):
    torch.manual_seed(SEED)
    a = torch.randn((params.m, params.k), dtype=torch.float16, device="cuda")
    b = torch.randn((params.k, params.n), dtype=torch.float16, device="cuda")
    bias = torch.randn((params.n,), dtype=torch.float32, device="cuda")
    output = torch.empty((params.m, params.n), dtype=torch.float32, device="cuda")
    cuda_all_inputs = [a, b, bias, output, params.m, params.k, params.n]
    torch_all_inputs = [a, b, bias]
    return cuda_all_inputs, torch_all_inputs, [output]


def cuda_output_tensor_transform(tensor):
    return tensor
