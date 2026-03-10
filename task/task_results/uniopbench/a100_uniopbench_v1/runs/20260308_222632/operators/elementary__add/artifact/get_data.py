import torch
import ctypes
from dataclasses import dataclass
from optest.tools.checker import SEED


@dataclass
class Params:
    """Add 参数配置"""

    shape: tuple = (18, 128)
    total_elements: int = 2304  # 18 * 128


def get_cuda_argtypes():
    return [
        ctypes.c_void_p,  # A (GPU pointer)
        ctypes.c_void_p,  # B (GPU pointer)
        ctypes.c_void_p,  # output (GPU pointer)
        ctypes.c_int,  # size
    ]


def get_cuda_torch_inputs(params: Params):
    torch.manual_seed(SEED)
    # Create data directly on specified device
    A = torch.randn(params.shape, dtype=torch.float32, device="cuda").normal_(
        mean=0.0, std=0.5
    )
    B = torch.randn(params.shape, dtype=torch.float32, device="cuda").normal_(
        mean=0.0, std=0.5
    )

    # Create output tensor
    output_cuda = torch.empty_like(A)
    cuda_output_tensors = [output_cuda]

    cuda_all_inputs = [A, B, output_cuda, params.total_elements]

    # For PyTorch, inputs are already in correct format
    torch_all_inputs = [A, B]
    return cuda_all_inputs, torch_all_inputs, cuda_output_tensors


def cuda_output_tensor_transform(cuda_output):
    return cuda_output
