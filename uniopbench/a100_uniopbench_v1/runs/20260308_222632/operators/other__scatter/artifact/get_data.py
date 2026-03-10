import torch
import ctypes
from dataclasses import dataclass
from optest.tools.checker import SEED


@dataclass
class Params:
    shape: tuple = (8, 768, 1, 1)
    N: int = 8
    C: int = 768
    H: int = 1
    W: int = 1


def get_cuda_argtypes():
    return [
        ctypes.c_void_p,  # input
        ctypes.c_void_p,  # indices
        ctypes.c_void_p,  # output
        ctypes.c_int,  # N
        ctypes.c_int,  # C
        ctypes.c_int,  # H
        ctypes.c_int,  # W
    ]


def get_cuda_torch_inputs(params: Params):
    torch.manual_seed(SEED)
    x = torch.randn(params.shape, dtype=torch.float32, device="cuda")
    rows = params.N * params.C * params.H
    perms = [
        torch.randperm(params.W, device="cuda", dtype=torch.int32) for _ in range(rows)
    ]
    indices = torch.stack(perms, dim=0).view(params.shape)
    output = x.clone()
    cuda_all_inputs = [x, indices, output, params.N, params.C, params.H, params.W]
    torch_all_inputs = [x, indices]
    return cuda_all_inputs, torch_all_inputs, [output]


def cuda_output_tensor_transform(tensor):
    return tensor
