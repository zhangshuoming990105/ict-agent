"""Manual test to debug the CUDA kernel"""
import torch
import sys
import os

# Add paths
sys.path.insert(0, os.path.dirname(__file__))

from get_data import get_cuda_torch_inputs, Params, get_cuda_argtypes, cuda_output_tensor_transform
from torch_.ref import torch_kernel
from optest.tools.builder import load_cuda_kernel

# Simple test parameters
class SimpleParams:
    batch_size = 1
    input_height = 5
    input_width = 5
    channels = 2
    kernel_height = 3
    kernel_width = 3
    stride = 1
    output_height = 3
    output_width = 3

params = SimpleParams()

# Load CUDA kernel
print("Loading CUDA kernel...")
argtypes = get_cuda_argtypes()
cuda_kernel = load_cuda_kernel(os.path.dirname(__file__), argtypes)

# Generate test data
torch.manual_seed(42)
x_nhwc = torch.arange(params.batch_size * params.input_height * params.input_width * params.channels, 
                      dtype=torch.float32).reshape(params.batch_size, params.input_height, params.input_width, params.channels).cuda()

output_shape_nhwc = (params.batch_size, params.output_height, params.output_width, params.channels)
output_cuda_nhwc = torch.zeros(output_shape_nhwc, dtype=torch.float32, device='cuda')

print(f"\nInput shape (NHWC): {x_nhwc.shape}")
print(f"Output shape (NHWC): {output_cuda_nhwc.shape}")
print(f"\nInput (channel 0):")
print(x_nhwc[0, :, :, 0].cpu())

# Run CUDA kernel
cuda_kernel(
    x_nhwc.data_ptr(),
    output_cuda_nhwc.data_ptr(),
    params.batch_size,
    params.channels,
    params.input_height,
    params.kernel_height,
    params.stride
)

print(f"\nCUDA Output (NHWC, channel 0):")
print(output_cuda_nhwc[0, :, :, 0].cpu())

# PyTorch reference
from optest.tools.layout import convert_nhwc_to_nchw
x_nchw = convert_nhwc_to_nchw(x_nhwc)
expected_nchw = torch_kernel(x_nchw, params.kernel_height, params.stride)

print(f"\nExpected (NCHW, channel 0):")
print(expected_nchw[0, 0, :, :].cpu())

# Convert CUDA output to NCHW for comparison
output_cuda_nchw = cuda_output_tensor_transform(output_cuda_nhwc)

print(f"\nCUDA Output (converted to NCHW, channel 0):")
print(output_cuda_nchw[0, 0, :, :].cpu())

print(f"\nDifference:")
diff = (output_cuda_nchw - expected_nchw).abs()
print(diff[0, 0, :, :].cpu())
print(f"Max error: {diff.max().item()}")
