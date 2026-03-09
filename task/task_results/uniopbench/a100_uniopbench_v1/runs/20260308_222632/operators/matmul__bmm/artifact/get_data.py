import torch
import ctypes
from dataclasses import dataclass
from optest.tools.builder import SEED


@dataclass
class Params:
    """BMM (Batch Matrix Multiplication) operation parameters"""

    batch_size: int = 4
    dim1: int = 512  # A.shape = (batch_size, dim1, dim2)
    dim2: int = 512  # B.shape = (batch_size, dim2, dim3)
    dim3: int = 512  # C.shape = (batch_size, dim1, dim3)


def get_cuda_argtypes():
    return [
        ctypes.c_void_p,  # A (half* GPU pointer)
        ctypes.c_void_p,  # B (half* GPU pointer)
        ctypes.c_void_p,  # C (float* GPU pointer)
        ctypes.c_int,  # b (batch_size)
        ctypes.c_int,  # m (dim1)
        ctypes.c_int,  # k (dim2)
        ctypes.c_int,  # n (dim3)
    ]


def get_cuda_torch_inputs(params: Params):
    torch.manual_seed(SEED)
    A_half = torch.randn(
        (params.batch_size, params.dim1, params.dim2),
        dtype=torch.float16,
        device="cuda",
    ).normal_(mean=0.0, std=0.5)
    B_half = torch.randn(
        (params.batch_size, params.dim2, params.dim3),
        dtype=torch.float16,
        device="cuda",
    ).normal_(mean=0.0, std=0.5)
    C = torch.empty(
        (params.batch_size, params.dim1, params.dim3),
        dtype=torch.float32,
        device="cuda",
    )

    # Convert to float32 for PyTorch comparison
    A_float32 = A_half.float()
    B_float32 = B_half.float()

    cuda_output_tensors = [C]

    cuda_all_inputs = [
        A_half,
        B_half,
        C,
        params.batch_size,  # b
        params.dim1,  # m
        params.dim2,  # k
        params.dim3,  # n
    ]
    torch_all_inputs = [A_float32, B_float32]
    return cuda_all_inputs, torch_all_inputs, cuda_output_tensors


def cuda_output_tensor_transform(cuda_output):
    return cuda_output  # No transformation needed for BMM
