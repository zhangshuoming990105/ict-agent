import torch
import ctypes
import os
import sys
from dataclasses import dataclass
from optest.tools.builder import SEED, load_cuda_kernel


@dataclass
class Params:
    """Softmax 参数配置"""

    shape: tuple = (7, 1, 6, 7)
    total_elements: int = 294  # 7 * 1 * 6 * 7
    dim: int = -1  # last dimension for softmax


def get_cuda_argtypes():
    return [
        ctypes.c_void_p,  # A (input GPU pointer)
        ctypes.c_void_p,  # C (output GPU pointer)
        ctypes.c_int,  # size1 (batch_size * seq_len = 12)
        ctypes.c_int,  # size2 (feature_dim = 5)
    ]


def get_cuda_torch_inputs(params: Params):
    """当需要cuda和torch比较时候使用"""
    torch.manual_seed(SEED)
    # Create data directly on specified device
    x = torch.randn(params.shape, dtype=torch.float32, device="cuda").normal_(
        mean=0.0, std=0.5
    )

    # Create output tensor
    output_cuda = torch.empty_like(x)
    cuda_output_tensors = [output_cuda]

    # For 4D tensor (7, 1, 6, 7), reshape to 2D treating last dim as feature
    # Flatten all dims except last: (7*1*6, 7) = (42, 7)
    original_shape = params.shape
    feature_dim = original_shape[-1]  # 7
    batch_elements = 1
    for dim in original_shape[:-1]:
        batch_elements *= dim  # 7*1*6 = 42

    x_2d = x.view(batch_elements, feature_dim)
    output_cuda_2d = output_cuda.view(batch_elements, feature_dim)

    size1 = batch_elements  # 42
    size2 = feature_dim  # 7
    cuda_all_inputs = [x_2d, output_cuda_2d, size1, size2]

    # For PyTorch, inputs are already in correct format (keep original 3D shape)
    torch_all_inputs = [x]
    return cuda_all_inputs, torch_all_inputs, cuda_output_tensors


def cuda_output_tensor_transform(cuda_output):
    # Triton kernel outputs 2D tensor, but PyTorch outputs 3D tensor
    # Reshape triton output back to original 3D shape to match PyTorch format for comparison
    # Get the original shape from the Params class
    params = Params()
    return cuda_output.view(params.shape)
