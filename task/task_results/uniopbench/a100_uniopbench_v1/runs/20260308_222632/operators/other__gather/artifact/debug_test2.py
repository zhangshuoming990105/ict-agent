import torch
import ctypes
from optest.tools.checker import SEED

# Replicate the exact test case with CUDA kernel
torch.manual_seed(SEED)
data_shape = (50, 128, 4)
num_indices = 4

params_tensor = torch.randn(data_shape, dtype=torch.float32, device="cuda")
indices = torch.randint(0, data_shape[-1], (num_indices,), dtype=torch.int64, device="cuda")
output = torch.empty((data_shape[0], data_shape[1], num_indices), dtype=torch.float32, device="cuda")

print(f"params shape: {params_tensor.shape}")
print(f"indices shape: {indices.shape}")
print(f"indices values: {indices}")
print(f"output shape: {output.shape}")

# Load the CUDA library
lib = ctypes.CDLL("cuda_/lib_cuda_kernel.so")

# Set up argument types
lib.cuda_kernel.argtypes = [
    ctypes.c_void_p,  # params
    ctypes.c_void_p,  # indices
    ctypes.c_void_p,  # output
    ctypes.c_int,  # dim0
    ctypes.c_int,  # dim1
    ctypes.c_int,  # dim2
    ctypes.c_int,  # num_indices
]

# Call CUDA kernel
lib.cuda_kernel(
    params_tensor.data_ptr(),
    indices.data_ptr(),
    output.data_ptr(),
    data_shape[0],
    data_shape[1],
    data_shape[2],
    num_indices
)

torch.cuda.synchronize()

# Compare with PyTorch reference
from torch_.ref import torch_kernel
expected = torch_kernel(params_tensor, indices)

print(f"expected shape: {expected.shape}")
print(f"output shape: {output.shape}")
print(f"Match: {torch.allclose(output, expected, rtol=1e-5, atol=1e-5)}")
if not torch.allclose(output, expected, rtol=1e-5, atol=1e-5):
    diff = (output - expected).abs()
    print(f"Max diff: {diff.max()}")
    print(f"Mean diff: {diff.mean()}")
