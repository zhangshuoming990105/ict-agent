"""Verbose debug of test framework."""

import torch
import sys
import os
import inspect
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(__file__))

from optest import TestCase, TensorSpec, ScalarSpec, TestConfig, CUDABackend
from torch_.ref import torch_kernel

@dataclass
class MaxPoolParams:
    batch_size: int = 4
    channels: int = 64
    height: int = 8
    width: int = 8
    kernel_size: int = 5
    stride: int = 3
    output_height: int = 2
    output_width: int = 2

testcase = TestCase(
    tensor_specs=[
        TensorSpec("input", torch.float32, ("batch_size", "channels", "height", "width"), "input"),
        TensorSpec("output", torch.float32, ("batch_size", "channels", "output_height", "output_width"), "output"),
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

print("=" * 80)
print("DATA GENERATED")
print("=" * 80)
print(f"all_inputs: {len(data['all_inputs'])} items")
for i, item in enumerate(data['all_inputs']):
    if isinstance(item, torch.Tensor):
        print(f"  [{i}] Tensor shape={item.shape}, stride={item.stride()}, device={item.device}")
    else:
        print(f"  [{i}] {type(item).__name__} = {item}")

print(f"\ntorch_inputs: {len(data['torch_inputs'])} items")
for i, item in enumerate(data['torch_inputs']):
    print(f"  [{i}] Tensor shape={item.shape}")

print(f"\noutputs: {len(data['outputs'])} items")
for i, item in enumerate(data['outputs']):
    print(f"  [{i}] Tensor shape={item.shape}, stride={item.stride()}")

# Run PyTorch reference with proper argument injection (mimicking runner.py)
print("\n" + "=" * 80)
print("PYTORCH REFERENCE (with arg injection)")
print("=" * 80)
ref_inputs = list(data['torch_inputs'])
print(f"Initial ref_inputs: {len(ref_inputs)}")

expected_args = len(inspect.signature(torch_kernel).parameters)
print(f"Expected args for torch_kernel: {expected_args}")

if expected_args > len(ref_inputs):
    float_scalars = []
    for spec in testcase.scalar_specs:
        if spec.dtype in (float, int):
            value = spec.value if not callable(spec.value) else spec.value(params)
            float_scalars.append(value)
    needed = expected_args - len(ref_inputs)
    ref_inputs = ref_inputs + float_scalars[:needed]
    print(f"Added {needed} scalar args: {float_scalars[:needed]}")

print(f"Final ref_inputs: {len(ref_inputs)}")
for i, item in enumerate(ref_inputs):
    if isinstance(item, torch.Tensor):
        print(f"  [{i}] Tensor shape={item.shape}")
    else:
        print(f"  [{i}] {type(item).__name__} = {item}")

output_ref = torch_kernel(*ref_inputs)
print(f"\nPyTorch output shape: {output_ref.shape}")
print(f"PyTorch output first 8 values: {output_ref.flatten()[:8].cpu().tolist()}")

# Run CUDA kernel
print("\n" + "=" * 80)
print("CUDA KERNEL")
print("=" * 80)
backend = CUDABackend()
kernel = backend.load_kernel(".", testcase.get_argtypes())
test_inputs = backend.prepare_inputs(data['all_inputs'])
print(f"Calling cuda_kernel with {len(test_inputs)} inputs (after prepare_inputs)")
backend.execute(kernel, test_inputs)
output_test = data['outputs'][0]
print(f"CUDA output shape: {output_test.shape}")
print(f"CUDA output first 8 values: {output_test.flatten()[:8].cpu().tolist()}")

# Compare
print("\n" + "=" * 80)
print("COMPARISON")
print("=" * 80)
print(f"output_ref shape: {output_ref.shape}, elements: {output_ref.numel()}")
print(f"output_test shape: {output_test.shape}, elements: {output_test.numel()}")
print(f"Shapes match: {output_ref.shape == output_test.shape}")

if output_ref.shape == output_test.shape:
    diff = (output_ref - output_test).abs()
    print(f"Max abs diff: {diff.max().item()}")
    print(f"Mean abs diff: {diff.mean().item()}")
    print(f"Match: {torch.allclose(output_ref, output_test, atol=1e-5)}")
else:
    print("ERROR: Shapes don't match!")
