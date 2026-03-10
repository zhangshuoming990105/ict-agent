import torch

def torch_kernel(input_tensor: torch.Tensor) -> torch.Tensor:
    # 按照 LeetCUDA 逻辑：输出大小为 max(a) + 1
    if input_tensor.numel() == 0:
        return torch.zeros(0, dtype=torch.int32, device=input_tensor.device)
    
    max_val = torch.max(input_tensor).item()
    # torch.bincount 是最贴近 LeetCUDA 逻辑的参考实现
    return torch.bincount(input_tensor, minlength=max_val + 1).to(torch.int32)