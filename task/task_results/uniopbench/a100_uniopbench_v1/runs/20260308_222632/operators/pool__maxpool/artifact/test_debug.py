"""Debug script to check tensor layout in optest framework."""

import os
import sys
import torch
from dataclasses import dataclass

# Add current dir to path
sys.path.insert(0, os.path.dirname(__file__))

from optest import TestCase, TensorSpec, ScalarSpec, TestConfig, CUDABackend
from torch_.ref import torch_kernel

@dataclass
class MaxPoolParams:
    """MaxPool test parameters."""
    batch_size: int = 4
    channels: int = 64
    height: int = 8
    width: int = 8
    kernel_size: int = 5
    stride: int = 3
    output_height: int = 2
    output_width: int = 2

# Define test case
testcase = TestCase(
    tensor_specs=[
        TensorSpec(
            "input",
            torch.float32,
            ("batch_size", "channels", "height", "width"),
            "input",
        ),
        TensorSpec(
            "output",
            torch.float32,
            ("batch_size", "channels", "output_height", "output_width"),
            "output",
        ),
    ],
    scalar_specs=[
        ScalarSpec("batch_size", int, lambda p: p.batch_size),
        ScalarSpec("channels", int, lambda p: p.channels),
        ScalarSpec("input_H", int, lambda p: p.height),
        ScalarSpec("kernel_size", int, lambda p: p.kernel_size),
        ScalarSpec("stride", int, lambda p: p.stride),
    ],
    torch_kernel=torch_kernel,
)

params = MaxPoolParams()
data = testcase.generate_data(params)

print("Generated data:")
print(f"  all_inputs length: {len(data['all_inputs'])}")
print(f"  torch_inputs length: {len(data['torch_inputs'])}")
print(f"  outputs length: {len(data['outputs'])}")

print("\nInput tensor:")
input_tensor = data['all_inputs'][0]
print(f"  Shape: {input_tensor.shape}")
print(f"  Stride: {input_tensor.stride()}")
print(f"  Is contiguous: {input_tensor.is_contiguous()}")
print(f"  Is channels_last: {input_tensor.is_contiguous(memory_format=torch.channels_last)}")

print("\nOutput tensor:")
output_tensor = data['all_inputs'][1]
print(f"  Shape: {output_tensor.shape}")
print(f"  Stride: {output_tensor.stride()}")
print(f"  Is contiguous: {output_tensor.is_contiguous()}")
print(f"  Is channels_last: {output_tensor.is_contiguous(memory_format=torch.channels_last)}")

print("\nScalar inputs:")
for i, val in enumerate(data['all_inputs'][2:], 2):
    print(f"  [{i}] = {val}")
