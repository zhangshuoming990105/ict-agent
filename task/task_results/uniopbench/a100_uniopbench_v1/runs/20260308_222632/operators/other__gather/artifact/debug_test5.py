"""Test indices generation multiple times."""

import torch
from optest.tools.checker import SEED

for trial in range(10):
    torch.manual_seed(SEED + trial)
    data_shape = (50, 128, 4)
    num_indices = 4
    
    indices = torch.randint(0, data_shape[-1], (num_indices,), dtype=torch.int64, device="cuda")
    
    print(f"Trial {trial}: indices = {indices.cpu().numpy()}, min={indices.min().item()}, max={indices.max().item()}")
    
    # Check if any are out of bounds
    if indices.min() < 0 or indices.max() >= data_shape[-1]:
        print(f"  WARNING: Out of bounds!")
