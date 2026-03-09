"""Debug comparison with NCHW layout."""

import torch
import ctypes

# Parameters
batch_size = 4
channels = 64
height = 8
width = 8
kernel_size = 5
stride = 3
output_height = 2
output_width = 2

print(f"Expected output shape: ({batch_size}, {channels}, {output_height}, {output_width})")

# Create test data in NCHW contiguous format (matching optest)
torch.manual_seed(42)
input_tensor = torch.randn((batch_size, channels, height, width), dtype=torch.float32, device="cuda")
output_cuda = torch.empty((batch_size, channels, output_height, output_width), dtype=torch.float32, device="cuda")

print(f"\nInput tensor:")
print(f"  Shape: {input_tensor.shape}")
print(f"  Stride: {input_tensor.stride()}")
print(f"  Is contiguous: {input_tensor.is_contiguous()}")

print(f"\nOutput tensor:")
print(f"  Shape: {output_cuda.shape}")
print(f"  Stride: {output_cuda.stride()}")
print(f"  Is contiguous: {output_cuda.is_contiguous()}")

# PyTorch reference
output_torch = torch.nn.functional.max_pool2d(input_tensor, kernel_size=kernel_size, stride=stride)
print(f"\nPyTorch output:")
print(f"  Shape: {output_torch.shape}")
print(f"  First 8 values: {output_torch.flatten()[:8].cpu().tolist()}")

# Call CUDA kernel
lib = ctypes.CDLL("./cuda_/lib_cuda_kernel.so")
lib.cuda_kernel.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
    ctypes.c_int, ctypes.c_int, ctypes.c_int
]

lib.cuda_kernel(
    input_tensor.data_ptr(),
    output_cuda.data_ptr(),
    batch_size, channels, height, kernel_size, stride
)
torch.cuda.synchronize()

print(f"\nCUDA output:")
print(f"  Shape: {output_cuda.shape}")
print(f"  First 8 values: {output_cuda.flatten()[:8].cpu().tolist()}")

# Compare
diff = (output_cuda - output_torch).abs()
print(f"\nDifference:")
print(f"  Max abs error: {diff.max().item()}")
print(f"  Mean abs error: {diff.mean().item()}")
print(f"  Match: {torch.allclose(output_cuda, output_torch, atol=1e-5, rtol=1e-5)}")

# Show actual differences
if diff.max() > 1e-5:
    print(f"\nLargest differences:")
    flat_diff = diff.flatten()
    top_indices = flat_diff.argsort(descending=True)[:5]
    for i in top_indices:
        idx = i.item()
        print(f"  Index {idx}: torch={output_torch.flatten()[idx].item():.6f}, cuda={output_cuda.flatten()[idx].item():.6f}, diff={flat_diff[idx].item():.6f}")
