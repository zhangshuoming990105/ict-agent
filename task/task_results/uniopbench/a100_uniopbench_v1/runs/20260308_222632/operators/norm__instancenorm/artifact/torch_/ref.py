import torch
import torch.nn.functional as F


def torch_kernel(
    x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor
) -> torch.Tensor:
    return F.instance_norm(x, weight=gamma, bias=beta, eps=1e-5)
