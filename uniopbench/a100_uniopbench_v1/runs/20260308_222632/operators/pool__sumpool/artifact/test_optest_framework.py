"""Debug the optest framework behavior"""
import os
import sys
import torch
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(__file__))

from optest import (
    TestCase,
    TensorSpec,
    ScalarSpec,
    TestConfig,
    CUDABackend,
    run_test,
)
from torch_.ref import torch_kernel
from optest.tools.layout import convert_nhwc_to_nchw

TESTCASE_DIR = os.path.dirname(__file__)

@dataclass
class SumPoolParams:
    """SumPool test parameters."""
    batch_size: int = 16
    input_height: int = 64
    input_width: int = 64
    channels: int = 64
    kernel_height: int = 5
    kernel_width: int = 5
    stride: int = 1
    output_height: int = 60
    output_width: int = 60

# Define test case
testcase = TestCase(
    tensor_specs=[
        TensorSpec(
            "input",
            torch.float32,
            ("batch_size", "input_height", "input_width", "channels"),
            "input",
        ),
        TensorSpec(
            "output",
            torch.float32,
            ("batch_size", "output_height", "output_width", "channels"),
            "output",
        ),
    ],
    scalar_specs=[
        ScalarSpec("batch_size", int, lambda p: p.batch_size),
        ScalarSpec("channels", int, lambda p: p.channels),
        ScalarSpec("input_H", int, lambda p: p.input_height),
        ScalarSpec("kernel_size", int, lambda p: p.kernel_height),
        ScalarSpec("stride", int, lambda p: p.stride),
    ],
    torch_kernel=torch_kernel,
    output_transform=convert_nhwc_to_nchw,
)

# Create test configuration
config = TestConfig(
    enable_perf=False,
    compile_only=False,
)

# Create backend
backend = CUDABackend()

# Create test parameters
params = SumPoolParams()

print(f"Parameters:")
print(f"  batch_size: {params.batch_size}")
print(f"  input: ({params.input_height}, {params.input_width})")
print(f"  channels: {params.channels}")
print(f"  kernel: {params.kernel_height}")
print(f"  stride: {params.stride}")
print(f"  output: ({params.output_height}, {params.output_width})")

# Run test with verbose
result = run_test(
    testcase_dir=TESTCASE_DIR,
    testcase=testcase,
    backend=backend,
    params=params,
    config=config,
)

print(f"\nTest result: {result.passed}")
if hasattr(result, 'validation'):
    print(f"Validation: {result.validation}")
if hasattr(result, 'ref_output'):
    print(f"Ref output shape: {result.ref_output.shape if isinstance(result.ref_output, torch.Tensor) else 'not a tensor'}")
    if isinstance(result.ref_output, torch.Tensor):
        print(f"Ref output sample: {result.ref_output.flatten()[:4]}")
if hasattr(result, 'test_output'):
    print(f"Test output shape: {result.test_output.shape if isinstance(result.test_output, torch.Tensor) else 'not a tensor'}")
    if isinstance(result.test_output, torch.Tensor):
        print(f"Test output sample: {result.test_output.flatten()[:4]}")
