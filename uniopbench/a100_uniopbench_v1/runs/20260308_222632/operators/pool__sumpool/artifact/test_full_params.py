"""Test with full parameters"""
import torch
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from get_data import get_cuda_torch_inputs, Params, get_cuda_argtypes, cuda_output_tensor_transform
from torch_.ref import torch_kernel
from optest.tools.builder import load_cuda_kernel, SEED

# Use full test parameters
params = Params()
print(f"Test parameters:")
print(f"  Input: ({params.batch_size}, {params.input_height}, {params.input_width}, {params.channels}) NHWC")
print(f"  Kernel: {params.kernel_height}, Stride: {params.stride}")
print(f"  Output: ({params.batch_size}, {params.output_height}, {params.output_width}, {params.channels}) NHWC")

# Load CUDA kernel
print("\nLoading CUDA kernel...")
argtypes = get_cuda_argtypes()
cuda_kernel = load_cuda_kernel(os.path.dirname(__file__), argtypes)

# Get test inputs using the same function as the test script
cuda_all_inputs, torch_all_inputs, cuda_output_tensors = get_cuda_torch_inputs(params)

print(f"\nCUDA inputs: {len(cuda_all_inputs)}")
for i, inp in enumerate(cuda_all_inputs):
    if isinstance(inp, torch.Tensor):
        print(f"  [{i}] Tensor: {inp.shape} {inp.dtype}")
    else:
        print(f"  [{i}] Scalar: {inp}")

print(f"\nTorch inputs: {len(torch_all_inputs)}")
for i, inp in enumerate(torch_all_inputs):
    if isinstance(inp, torch.Tensor):
        print(f"  [{i}] Tensor: {inp.shape} {inp.dtype}")
    else:
        print(f"  [{i}] Scalar: {inp}")

# Run CUDA kernel
print("\nRunning CUDA kernel...")
x_nhwc = cuda_all_inputs[0]
output_cuda_nhwc = cuda_all_inputs[1]

cuda_kernel(
    x_nhwc.data_ptr(),
    output_cuda_nhwc.data_ptr(),
    params.batch_size,
    params.channels,
    params.input_height,
    params.kernel_height,
    params.stride
)

print(f"CUDA output shape: {output_cuda_nhwc.shape}")
print(f"CUDA output sample (first 4 from first batch, first spatial position, all channels):")
print(output_cuda_nhwc[0, 0, 0, :4])

# Run PyTorch reference
print("\nRunning PyTorch reference...")
x_nchw = torch_all_inputs[0]
expected_nchw = torch_kernel(*torch_all_inputs)

print(f"PyTorch output shape: {expected_nchw.shape}")
print(f"PyTorch output sample (first 4 channels from first batch, first spatial position):")
print(expected_nchw[0, :4, 0, 0])

# Convert CUDA output for comparison
print("\nConverting CUDA output to NCHW...")
output_cuda_nchw = cuda_output_tensor_transform(output_cuda_nhwc)

print(f"CUDA output (converted) shape: {output_cuda_nchw.shape}")
print(f"CUDA output (converted) sample:")
print(output_cuda_nchw[0, :4, 0, 0])

# Compare
print("\nComparing...")
diff = (output_cuda_nchw - expected_nchw).abs()
print(f"Max error: {diff.max().item()}")
print(f"Mean error: {diff.mean().item()}")

# Show where errors occur
if diff.max().item() > 0.01:
    print("\nLargest errors at (first few):")
    flat_diff = diff.flatten()
    top_indices = torch.topk(flat_diff, min(5, flat_diff.numel())).indices
    for idx in top_indices:
        print(f"  Index {idx.item()}: error = {flat_diff[idx].item()}")
