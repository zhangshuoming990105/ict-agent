import torch
import ctypes
from dataclasses import dataclass
from optest.tools.checker import SEED


@dataclass
class Params:
    shape: tuple = (16, 3, 32, 32)
    batch_size: int = 16
    num_channels: int = 3
    spatial_size: int = 1024
    eps: float = 1e-5


def get_cuda_argtypes():
    return [
        ctypes.c_void_p,  # input
        ctypes.c_void_p,  # output
        ctypes.c_void_p,  # running mean
        ctypes.c_void_p,  # running variance
        ctypes.c_void_p,  # gamma
        ctypes.c_void_p,  # beta
        ctypes.c_int,  # batch size
        ctypes.c_int,  # num channels
        ctypes.c_int,  # spatial size
    ]


def get_cuda_torch_inputs(params: Params):
    torch.manual_seed(SEED)
    x = torch.randn(params.shape, dtype=torch.float32, device="cuda")
    running_mean = torch.randn(params.num_channels, dtype=torch.float32, device="cuda")
    running_var = (
        torch.rand(params.num_channels, dtype=torch.float32, device="cuda") + 0.5
    )
    gamma = torch.randn(params.num_channels, dtype=torch.float32, device="cuda")
    beta = torch.randn(params.num_channels, dtype=torch.float32, device="cuda")
    output = torch.empty_like(x)
    cuda_all_inputs = [
        x,
        output,
        running_mean,
        running_var,
        gamma,
        beta,
        params.batch_size,
        params.num_channels,
        params.spatial_size,
    ]
    torch_all_inputs = [x, running_mean, running_var, gamma, beta]
    return cuda_all_inputs, torch_all_inputs, [output]


def cuda_output_tensor_transform(tensor):
    return tensor
