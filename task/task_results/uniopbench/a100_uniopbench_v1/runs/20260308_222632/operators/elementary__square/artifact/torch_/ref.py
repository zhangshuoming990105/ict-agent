"""PyTorch reference implementation for Square operator."""

import torch


def torch_kernel(x: torch.Tensor) -> torch.Tensor:
    """Square operator: output = x ** 2.

    Args:
        x: Input tensor.

    Returns:
        Element-wise squared tensor.
    """
    return torch.mul(x, x)

