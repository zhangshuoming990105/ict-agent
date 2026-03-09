import torch
import ctypes
from dataclasses import dataclass
from optest.tools.checker import SEED


@dataclass
class Params:
    data_shape: tuple = (50, 128, 4)
    num_indices: int = 4


def get_cuda_argtypes():
    return [
        ctypes.c_void_p,  # params
        ctypes.c_void_p,  # indices
        ctypes.c_void_p,  # output
        ctypes.c_int,  # dim0
        ctypes.c_int,  # dim1
        ctypes.c_int,  # dim2
        ctypes.c_int,  # num_indices
    ]


def get_cuda_torch_inputs(params: Params):
    torch.manual_seed(SEED)
    params_tensor = torch.randn(params.data_shape, dtype=torch.float32, device="cuda")
    indices = torch.randint(
        0,
        params.data_shape[-1],
        (params.num_indices,),
        dtype=torch.int64,
        device="cuda",
    )
    output = torch.empty(
        (params.data_shape[0], params.data_shape[1], params.num_indices),
        dtype=torch.float32,
        device="cuda",
    )
    cuda_all_inputs = [
        params_tensor,
        indices,
        output,
        params.data_shape[0],
        params.data_shape[1],
        params.data_shape[2],
        params.num_indices,
    ]
    torch_all_inputs = [params_tensor, indices]
    return cuda_all_inputs, torch_all_inputs, [output]


def cuda_output_tensor_transform(tensor):
    return tensor
