import torch
import ctypes

# Load library
lib = ctypes.CDLL('./cuda_/lib_cuda_kernel.so')
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

print(f"Input:\n{x}")
print(f"Indices:\n{indices}")

lib.cuda_kernel(
    ctypes.c_void_p(x.data_ptr()),
    ctypes.c_void_p(indices.data_ptr()),
    ctypes.c_void_p(output.data_ptr()),
    N, C, H, W
)

torch.cuda.synchronize()
print(f"\nCUDA output:\n{output}")

# Now try PyTorch scatter
expected = x.clone()
expected.scatter_(dim=3, index=indices.to(torch.long), src=x)
print(f"\nPyTorch expected:\n{expected}")

print(f"\nMatch: {torch.allclose(output, expected)}")
if not torch.allclose(output, expected):
    print(f"Max diff: {(output - expected).abs().max().item()}")
