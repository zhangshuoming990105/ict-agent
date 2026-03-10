import torch
import ctypes
from dataclasses import dataclass
from optest.tools.checker import SEED


@dataclass
class Params:
    batch: int = 32
    k: int = 128
    n: int = 128


def get_cuda_argtypes():
    return [
        ctypes.c_void_p,  # X
        ctypes.c_void_p,  # A
        ctypes.c_void_p,  # B
        ctypes.c_void_p,  # output
        ctypes.c_int,  # batch
        ctypes.c_int,  # K
        ctypes.c_int,  # N
    ]


def get_cuda_torch_inputs(params: Params):
    torch.manual_seed(SEED)
    x = torch.randn((params.batch, params.k), dtype=torch.float16, device="cuda")
    a = torch.randn((params.k, params.n), dtype=torch.float16, device="cuda")
    b = torch.randn((params.k, params.n), dtype=torch.float16, device="cuda")
    output = torch.empty((params.batch, params.n), dtype=torch.float32, device="cuda")
    cuda_all_inputs = [x, a, b, output, params.batch, params.k, params.n]
    torch_all_inputs = [x, a, b]
    return cuda_all_inputs, torch_all_inputs, [output]


def cuda_output_tensor_transform(tensor):
    return tensor
