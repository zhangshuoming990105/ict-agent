import torch


def torch_kernel(x: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.elu(x, alpha=1.0)
