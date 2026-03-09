"""Debug script to understand test.py flow"""
import os
import sys
import torch
from dataclasses import dataclass
from optest.tools.checker import SEED

TESTCASE_DIR = os.path.dirname(__file__)
sys.path.insert(0, TESTCASE_DIR)

@dataclass
class GatherParams:
    """Gather test parameters."""
    data_shape: tuple = (50, 128, 4)
    num_indices: int = 4
    output_shape: tuple = (50, 128, 4)

# Create params
params = GatherParams()

# Generate data like optest does
torch.manual_seed(SEED)

# Create tensors based on TensorSpec
data_shape = params.data_shape
num_indices = params.num_indices

params_tensor = torch.randn(data_shape, dtype=torch.float32, device='cuda')
indices = torch.randint(0, data_shape[2], (num_indices,), dtype=torch.int64, device='cuda')
output = torch.empty((data_shape[0], data_shape[1], num_indices), dtype=torch.float32, device='cuda')

print('Generated tensors:')
print('  params_tensor:', params_tensor.shape, params_tensor.dtype)
print('  indices:', indices.shape, indices.dtype)
print('  output:', output.shape, output.dtype)
print('  indices values:', indices.cpu().numpy())

# Check what scalars are passed
dim0 = data_shape[0]
dim1 = data_shape[1]
dim2 = data_shape[2]
num_indices_scalar = num_indices

print('\nScalar args:')
print('  dim0:', dim0)
print('  dim1:', dim1)
print('  dim2:', dim2)
print('  num_indices:', num_indices_scalar)

# Test reference
from torch_.ref import torch_kernel
torch_output = torch_kernel(params_tensor, indices)
print('\nTorch output shape:', torch_output.shape)
print('Torch output[0,0,:]:', torch_output[0,0,:].cpu().numpy())

# Test CUDA kernel
from get_data import get_cuda_argtypes
from optest.core.backend import CUDABackend

backend = CUDABackend()
kernel = backend.load_kernel('.', get_cuda_argtypes())

print('\nCalling CUDA kernel...')
kernel(
    params_tensor.data_ptr(),
    indices.data_ptr(),
    output.data_ptr(),
    dim0,
    dim1,
    dim2,
    num_indices_scalar
)
torch.cuda.synchronize()
print('CUDA output shape:', output.shape)
print('CUDA output[0,0,:]:', output[0,0,:].cpu().numpy())

# Compare
print('\nMatch:', torch.allclose(torch_output, output, rtol=1e-5, atol=1e-5))
