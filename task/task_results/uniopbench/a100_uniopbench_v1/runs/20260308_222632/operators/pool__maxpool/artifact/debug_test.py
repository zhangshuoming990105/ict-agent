import torch
import ctypes
from test import MaxPoolParams

# Create params
params = MaxPoolParams()

print(f"Test parameters:")
print(f"  Input: {params.batch_size}x{params.channels}x{params.height}x{params.width}")
print(f"  Output: {params.batch_size}x{params.channels}x{params.output_height}x{params.output_width}")
print(f"  Kernel: {params.kernel_size}, Stride: {params.stride}")

# Create tensors
torch.manual_seed(42)
x = torch.randn(params.batch_size, params.channels, params.height, params.width, device='cuda')
x = x.to(memory_format=torch.channels_last)

y = torch.empty(params.batch_size, params.channels, params.output_height, params.output_width, device='cuda')
y = y.to(memory_format=torch.channels_last)

print(f"\nInput tensor: shape={x.shape}, stride={x.stride()}, is_channels_last={x.is_contiguous(memory_format=torch.channels_last)}")
print(f"Output tensor: shape={y.shape}, stride={y.stride()}, is_channels_last={y.is_contiguous(memory_format=torch.channels_last)}")

# Load and call CUDA kernel
lib = ctypes.CDLL('cuda_/lib_cuda_kernel.so')
lib.cuda_kernel.argtypes = [
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
]

lib.cuda_kernel(
    x.data_ptr(),
    y.data_ptr(),
    params.batch_size,
    params.channels,
    params.height,
    params.kernel_size,
    params.stride,
)
torch.cuda.synchronize()

# Compute reference
ref = torch.nn.functional.max_pool2d(x, kernel_size=params.kernel_size, stride=params.stride)

print(f"\nReference output: shape={ref.shape}")
print(f"CUDA output: shape={y.shape}")

# Check correctness
diff = (ref - y).abs()
max_diff = diff.max().item()
mean_diff = diff.mean().item()

print(f"\nMax absolute error: {max_diff}")
print(f"Mean absolute error: {mean_diff}")

if max_diff < 1e-5:
    print("\n✅ PASSED")
else:
    print("\n❌ FAILED")
    print(f"\nFirst mismatch (expected vs actual):")
    flat_ref = ref.flatten()
    flat_y = y.flatten()
    for i in range(min(10, len(flat_ref))):
        if abs(flat_ref[i] - flat_y[i]) > 1e-5:
            print(f"  Index {i}: {flat_ref[i]:.6f} vs {flat_y[i]:.6f}")
