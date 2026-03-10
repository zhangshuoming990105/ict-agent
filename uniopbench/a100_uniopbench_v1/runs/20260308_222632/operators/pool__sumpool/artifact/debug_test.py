import torch
import numpy as np
from optest.tools.layout import convert_nhwc_to_nchw

# Simple test case
batch = 1
h = 5
w = 5
c = 2
kernel_size = 3
stride = 1
output_h = (h - kernel_size) // stride + 1  # = 3
output_w = (w - kernel_size) // stride + 1  # = 3

print(f"Input: ({batch}, {h}, {w}, {c}) NHWC")
print(f"Output: ({batch}, {output_h}, {output_w}, {c}) NHWC")
print(f"Kernel: {kernel_size}, Stride: {stride}")

# Create simple input in NHWC format
torch.manual_seed(42)
x_nhwc = torch.arange(batch * h * w * c, dtype=torch.float32).reshape(batch, h, w, c)

print("\nInput tensor (NHWC) - channel 0:")
print(x_nhwc[0, :, :, 0])

# PyTorch reference
x_nchw = convert_nhwc_to_nchw(x_nhwc)
print("\nInput tensor (NCHW) - channel 0:")
print(x_nchw[0, 0, :, :])

expected = torch.nn.functional.avg_pool2d(x_nchw, kernel_size=kernel_size, stride=stride) * (kernel_size * kernel_size)
print("\nPyTorch Output (NCHW) - channel 0:")
print(expected[0, 0, :, :])

# Manual calculation for verification
print("\nManual sum pooling calculation for position (0,0) channel 0:")
print("Input window (0:3, 0:3):")
print(x_nchw[0, 0, 0:3, 0:3])
print(f"Sum: {x_nchw[0, 0, 0:3, 0:3].sum().item()}")
print(f"Expected at (0,0): {expected[0, 0, 0, 0].item()}")
