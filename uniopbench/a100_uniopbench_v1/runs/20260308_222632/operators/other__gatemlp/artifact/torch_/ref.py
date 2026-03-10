import torch
import torch.nn.functional as F


def torch_kernel(x: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    x64 = x.float().double()
    o1 = torch.matmul(x64, a.float().double())
    o2 = torch.matmul(x64, b.float().double())
    return (F.silu(o1) * o2).float()
