import torch
from torch_.ref import torch_kernel
import ctypes

# Replicate exact test sequence
N, C, H, W = 8, 768, 1, 1

# Generate test data the same way
torch.manual_seed(42)  # Use SEED
x = torch.randn((N, C, H, W), dtype=torch.float32, device="cuda")
rows = N * C * H
perms = [torch.randperm(W, device="cuda", dtype=torch.int32) for _ in range(rows)]
indices = torch.stack(perms, dim=0).view((N, C, H, W))
output = x.clone()

print(f"Shape: {x.shape}")
print(f"Indices dtype: {indices.dtype}")

# 1. Run PyTorch reference
print("Running PyTorch reference...")
torch_output = torch_kernel(x, indices)
print("PyTorch reference completed")

# 2. Load and run CUDA kernel
lib = ctypes.CDLL('./cuda_/lib_cuda_kernel.so')
lib.cuda_kernel.argtypes = [
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
]

cuda_output = torch.zeros_like(x)
print("Running CUDA kernel...")
lib.cuda_kernel(
    ctypes.c_void_p(x.data_ptr()),
    ctypes.c_void_p(indices.data_ptr()),
    ctypes.c_void_p(cuda_output.data_ptr()),
    N, C, H, W
)

print("Synchronizing...")
torch.cuda.synchronize()
print("CUDA kernel completed")

# 3. Compare
print(f"Match: {torch.allclose(cuda_output, torch_output)}")
