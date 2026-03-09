import torch
from optest.tools.checker import SEED

# Replicate the exact test case
torch.manual_seed(SEED)
data_shape = (50, 128, 4)
num_indices = 4

params_tensor = torch.randn(data_shape, dtype=torch.float32, device="cuda")
indices = torch.randint(0, data_shape[-1], (num_indices,), dtype=torch.int64, device="cuda")

print(f"params shape: {params_tensor.shape}")
print(f"indices shape: {indices.shape}")
print(f"indices values: {indices}")

# Try the reference implementation
expanded_indices = indices.view(1, 1, -1).expand(params_tensor.size(0), params_tensor.size(1), -1)
print(f"expanded_indices shape: {expanded_indices.shape}")

try:
    result = torch.gather(params_tensor, dim=2, index=expanded_indices)
    print(f"result shape: {result.shape}")
    print("Success!")
except Exception as e:
    print(f"Error: {e}")
