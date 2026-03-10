import torch
import ctypes
import os

# Load the CUDA kernel
lib = ctypes.CDLL(os.path.join("cuda_", "lib_cuda_kernel.so"))
lib.cuda_kernel.argtypes = [
    ctypes.c_void_p,  # input
    ctypes.c_void_p,  # indices
    ctypes.c_void_p,  # output
    ctypes.c_int,  # N
    ctypes.c_int,  # C
    ctypes.c_int,  # H
    ctypes.c_int,  # W
]

# Create simple test data
N, C, H, W = 2, 3, 2, 4
x = torch.arange(N*C*H*W, dtype=torch.float32, device="cuda").reshape(N, C, H, W)
# Create indices that map each position to itself (identity scatter)
indices = torch.arange(W, dtype=torch.int32, device="cuda").unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(N, C, H, W)
print("Input shape:", x.shape)
print("Indices shape:", indices.shape)
print("Input:\n", x)
print("Indices:\n", indices)

output = torch.zeros_like(x)

# Call CUDA kernel
lib.cuda_kernel(
    ctypes.c_void_p(x.data_ptr()),
    ctypes.c_void_p(indices.data_ptr()),
    ctypes.c_void_p(output.data_ptr()),
    N, C, H, W
)

torch.cuda.synchronize()

print("Output:\n", output)

# Expected: output should equal x since we're scattering to the same positions
print("Are they equal?", torch.allclose(x, output))
