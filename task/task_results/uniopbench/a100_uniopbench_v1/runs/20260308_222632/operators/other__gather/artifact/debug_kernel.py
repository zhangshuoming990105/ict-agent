import sys
import os
import torch
import ctypes
from optest.tools.checker import SEED

sys.path.insert(0, '.')

# Get argtypes
from get_data import get_cuda_argtypes

# Load the kernel
from optest.core.backend import CUDABackend
backend = CUDABackend()
kernel = backend.load_kernel('.', get_cuda_argtypes())

# Generate test data
torch.manual_seed(SEED)
params_tensor = torch.randn((50, 128, 4), dtype=torch.float32, device='cuda')
indices = torch.randint(0, 4, (4,), dtype=torch.int64, device='cuda')
output = torch.empty((50, 128, 4), dtype=torch.float32, device='cuda')

print('Input shapes:')
print('  params:', params_tensor.shape, params_tensor.dtype)
print('  indices:', indices.shape, indices.dtype)
print('  output:', output.shape, output.dtype)
print('Indices:', indices.cpu().numpy())
print('Index range: [%d, %d]' % (indices.min().item(), indices.max().item()))

# Call kernel
print('\nCalling CUDA kernel...')
try:
    kernel(
        params_tensor.data_ptr(),
        indices.data_ptr(),
        output.data_ptr(),
        50,  # dim0
        128,  # dim1
        4,  # dim2
        4,  # num_indices
    )
    torch.cuda.synchronize()
    print('Kernel executed successfully!')
    print('Output shape:', output.shape)
    print('Output sample:', output[0, 0, :].cpu().numpy())
except Exception as e:
    print('Error:', e)
    import traceback
    traceback.print_exc()
