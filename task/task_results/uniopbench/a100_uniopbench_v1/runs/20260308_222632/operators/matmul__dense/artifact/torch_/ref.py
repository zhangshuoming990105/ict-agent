import torch


def torch_kernel(a: torch.Tensor, b: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    result = torch.matmul(a.float(), b.float())
    return result + bias
