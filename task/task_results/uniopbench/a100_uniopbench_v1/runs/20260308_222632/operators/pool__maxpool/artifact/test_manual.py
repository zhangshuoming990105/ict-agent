import torch
import ctypes
import os

# Parameters matching the test
batch_size = 4
channels = 64
height = 8
width = 8
kernel_size = 5
stride = 3
output_height = (height - kernel_size) // stride + 1  # Should be 2
output_width = (width - kernel_size) // stride + 1    # Should be 2

print(f"Input shape: ({batch_size}, {channels}, {height}, {width})")
print(f"Output shape should be: ({batch_size}, {channels}, {output_height}, {output_width})")

# Create test data
torch.manual_seed(42)
input_tensor = torch.randn((batch_size, channels, height, width), dtype=torch.float32, device="cuda")
input_tensor = input_tensor.to(memory_format=torch.channels_last)

print(f"\nInput tensor:")
print(f"  Shape: {input_tensor.shape}")
print(f"  Stride: {input_tensor.stride()}")
print(f"  Is channels_last: {input_tensor.is_contiguous(memory_format=torch.channels_last)}")

# PyTorch reference
output_torch = torch.nn.functional.max_pool2d(input_tensor, kernel_size=kernel_size, stride=stride)
print(f"\nPyTorch output:")
print(f"  Shape: {output_torch.shape}")
print(f"  Stride: {output_torch.stride()}")
print(f"  First few values: {output_torch.flatten()[:8].cpu().tolist()}")

# CUDA kernel
output_cuda = torch.empty((batch_size, channels, output_height, output_width), dtype=torch.float32, device="cuda")
output_cuda = output_cuda.to(memory_format=torch.channels_last)

print(f"\nCUDA output buffer:")
print(f"  Shape: {output_cuda.shape}")
print(f"  Stride: {output_cuda.stride()}")
print(f"  Is channels_last: {output_cuda.is_contiguous(memory_format=torch.channels_last)}")

# Load and call CUDA kernel
lib = ctypes.CDLL("./cuda_/lib_cuda_kernel.so")
lib.cuda_kernel.argtypes = [
    ctypes.c_void_p,  # input
    ctypes.c_void_p,  # output
    ctypes.c_int,     # batch_size
    ctypes.c_int,     # channels
    ctypes.c_int,     # input_H
    ctypes.c_int,     # kernel_size
    ctypes.c_int,     # stride
]

lib.cuda_kernel(
    input_tensor.data_ptr(),
    output_cuda.data_ptr(),
    batch_size,
    channels,
    height,
    kernel_size,
    stride
)

torch.cuda.synchronize()

print(f"\nCUDA output after kernel:")
print(f"  Shape: {output_cuda.shape}")
print(f"  First few values: {output_cuda.flatten()[:8].cpu().tolist()}")

# Compare
diff = (output_cuda - output_torch).abs()
print(f"\nComparison:")
print(f"  Max abs error: {diff.max().item()}")
print(f"  Mean abs error: {diff.mean().item()}")
print(f"  Match: {torch.allclose(output_cuda, output_torch, atol=1e-5, rtol=1e-5)}")
