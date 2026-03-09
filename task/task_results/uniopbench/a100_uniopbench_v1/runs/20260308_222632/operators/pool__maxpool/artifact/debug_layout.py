import torch
import ctypes

# Create a small test tensor
batch, channels, height, width = 1, 2, 4, 4
x = torch.zeros(batch, channels, height, width, dtype=torch.float32)

# Fill with unique values so we can track them
val = 0
for n in range(batch):
    for c in range(channels):
        for h in range(height):
            for w in range(width):
                x[n, c, h, w] = val
                val += 1

print("Original NCHW tensor:")
print(x)
print(f"Shape: {x.shape}")
print(f"Stride: {x.stride()}")

# Convert to channels_last
x_cl = x.to(memory_format=torch.channels_last)
print("\nChannels_last tensor (same logical view):")
print(x_cl)
print(f"Shape: {x_cl.shape}")
print(f"Stride: {x_cl.stride()}")

# Get actual memory via data_ptr
# Create a contiguous copy to see physical memory
x_cont = x_cl.contiguous(memory_format=torch.channels_last)

print("\nLet's manually print memory layout for channels_last:")
print("Stride explanation: (32, 1, 8, 2) means:")
print("  - batch: 32 elements apart")
print("  - channel: 1 element apart (adjacent)")
print("  - height: 8 elements apart")
print("  - width: 2 elements apart")
print("\nSo physical memory order is: N, H, W, C")
print("For position [n=0, h=0, w=0]: channels 0,1 are at indices 0,1")
print("For position [n=0, h=0, w=1]: channels 0,1 are at indices 2,3")
print("For position [n=0, h=1, w=0]: channels 0,1 are at indices 8,9")

# Verify with indexing
print("\nVerifying logical values at different positions:")
print(f"x_cl[0,0,0,0] = {x_cl[0,0,0,0].item()} (should be 0)")
print(f"x_cl[0,1,0,0] = {x_cl[0,1,0,0].item()} (should be 16)")
print(f"x_cl[0,0,0,1] = {x_cl[0,0,0,1].item()} (should be 1)")
print(f"x_cl[0,1,0,1] = {x_cl[0,1,0,1].item()} (should be 17)")

# To see actual memory, we need to use stride-aware indexing
print("\nPhysical memory addresses (using stride):")
base = 0
for h in range(2):  # Just first 2 rows
    for w in range(4):
        for c in range(2):
            mem_offset = 0 * x_cl.stride(0) + c * x_cl.stride(1) + h * x_cl.stride(2) + w * x_cl.stride(3)
            val = x_cl[0, c, h, w].item()
            print(f"[h={h}, w={w}, c={c}] -> mem_offset={mem_offset}, value={val}")
