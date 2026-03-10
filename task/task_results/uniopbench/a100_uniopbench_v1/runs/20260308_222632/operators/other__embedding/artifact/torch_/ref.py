import torch

def torch_kernel(indices: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.embedding(indices.long(), weight)