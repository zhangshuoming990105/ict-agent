import torch

# Test PyTorch's scatter_ to understand the expected behavior
N, C, H, W = 2, 3, 2, 4
x = torch.arange(N*C*H*W, dtype=torch.float32, device="cuda").reshape(N, C, H, W)
indices = torch.arange(W, dtype=torch.int64, device="cuda").unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(N, C, H, W)

print("Input shape:", x.shape)
print("Indices shape:", indices.shape)
print("Input:\n", x)
print("Indices:\n", indices)

output = torch.zeros_like(x)
output.scatter_(dim=3, index=indices, src=x)

print("PyTorch scatter output:\n", output)
print("Are they equal?", torch.allclose(x, output))
