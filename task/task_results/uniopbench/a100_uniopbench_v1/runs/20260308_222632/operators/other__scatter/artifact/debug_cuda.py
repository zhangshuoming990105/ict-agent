import torch
import ctypes
from cuda_ import load_cuda_kernel

# Load CUDA kernel
lib = load_cuda_kernel()
lib.cuda_kernel.argtypes = [
    ctypes.c_void_p,  # input
    ctypes.c_void_p,  # indices  
    ctypes.c_void_p,  # output
    ctypes.c_int,  # N
    ctypes.c_int,  # C
    ctypes.c_int,  # H
    ctypes.c_int,  # W
]

# Simple test
N, C, H, W = 2, 3, 1, 4
x = torch.randn(N, C, H, W, device='cuda')
indices = torch.stack([torch.randperm(W, device='cuda', dtype=torch.int32) for _ in range(N*C*H)]).view(N, C, H, W)
output = torch.zeros_like(x)

print(f"Before CUDA kernel:")
print(f"x: {x.data_ptr():x}")
print(f"indices: {indices.data_ptr():x}")
print(f"output: {output.data_ptr():x}")

lib.cuda_kernel(
    ctypes.c_void_p(x.data_ptr()),
    ctypes.c_void_p(indices.data_ptr()),
    ctypes.c_void_p(output.data_ptr()),
    N, C, H, W
)

torch.cuda.synchronize()
print(f"After CUDA kernel - success!")
print(f"output:\n{output}")

# Now try PyTorch scatter
try:
    expected = x.clone()
    expected.scatter_(dim=3, index=indices.to(torch.long), src=x)
    print(f"PyTorch scatter success!")
    print(f"expected:\n{expected}")
    
    print(f"\nMatch: {torch.allclose(output, expected)}")
except Exception as e:
    print(f"PyTorch scatter failed: {e}")
