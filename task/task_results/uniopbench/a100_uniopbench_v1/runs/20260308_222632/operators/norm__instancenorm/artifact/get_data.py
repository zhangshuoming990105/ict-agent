import torch
import ctypes
from dataclasses import dataclass
from optest.tools.checker import SEED


@dataclass
class Params:
    shape: tuple = (1, 3, 224, 224)
    batch: int = 1
    channels: int = 3
    spatial: int = 50176


def get_cuda_argtypes():
    return [
        ctypes.c_void_p,  # input
        ctypes.c_void_p,  # output
        ctypes.c_void_p,  # gamma
        ctypes.c_void_p,  # beta
        ctypes.c_int,  # batch
        ctypes.c_int,  # channels
        ctypes.c_int,  # spatial
    ]


def get_cuda_torch_inputs(params: Params):
    torch.manual_seed(SEED)
    x = torch.randn(params.shape, dtype=torch.float32, device="cuda")
    gamma = torch.randn(params.channels, dtype=torch.float32, device="cuda")
    beta = torch.randn(params.channels, dtype=torch.float32, device="cuda")
    output = torch.empty_like(x)
    cuda_all_inputs = [
        x,
        output,
        gamma,
        beta,
        params.batch,
        params.channels,
        params.spatial,
    ]
    torch_all_inputs = [x, gamma, beta]
    return cuda_all_inputs, torch_all_inputs, [output]


def cuda_output_tensor_transform(tensor):
    return tensor
