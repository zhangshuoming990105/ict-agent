"""Debug script to check tensor spec resolution."""

import os
import sys
import torch
from dataclasses import dataclass
from optest import (
    TestCase,
    TensorSpec,
    ScalarSpec,
)

from torch_.ref import torch_kernel

# Add parent directory to path for imports
TESTCASE_DIR = os.path.dirname(__file__)
sys.path.insert(0, TESTCASE_DIR)


@dataclass
class GatherParams:
    """Gather test parameters."""

    data_shape: tuple = (50, 128, 4)
    num_indices: int = 4
    output_shape: tuple = (50, 128, 4)


# Define test case using new framework
testcase = TestCase(
    tensor_specs=[
        TensorSpec("params", torch.float32, ("data_shape",), "input"),
        TensorSpec("indices", torch.int64, ("num_indices",), "input"),
        TensorSpec(
            "output",
            torch.float32,
            ("data_shape", 0, "data_shape", 1, "num_indices"),
            "output",
        ),
    ],
    scalar_specs=[
        ScalarSpec("dim0", int, lambda p: p.data_shape[0]),
        ScalarSpec("dim1", int, lambda p: p.data_shape[1]),
        ScalarSpec("dim2", int, lambda p: p.data_shape[2]),
        ScalarSpec("dim3", int, lambda p: p.num_indices),
    ],
    torch_kernel=torch_kernel,
)

params = GatherParams()

# Try to understand what shapes are being generated
from optest.execution.runner import TestRunner
from optest.core.backend import CUDABackend
from optest import TestConfig

backend = CUDABackend()
config = TestConfig(enable_perf=False, compile_only=False)

runner = TestRunner(TESTCASE_DIR, backend)

# Generate test inputs
test_inputs = runner._generate_test_inputs(testcase, params)

print("Generated test inputs:")
for name, tensor in test_inputs.items():
    if isinstance(tensor, torch.Tensor):
        print(f"  {name}: shape={tensor.shape}, dtype={tensor.dtype}")
    else:
        print(f"  {name}: {tensor}")
