import torch


def torch_kernel(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """A simple PyTorch implementation to verify the logic."""
    B, N_CTX, H, D_HEAD = q.shape

    # The operation is independent for each item in batch and sequence,
    # so we can use batch matrix multiplication.
    # Reshape to treat (B, N_CTX) as the batch dimension for matmul.
    q_b = q.view(B * N_CTX, H, D_HEAD)
    k_b = k.view(B * N_CTX, H, D_HEAD)
    v_b = v.view(B * N_CTX, H, D_HEAD)

    # S = Q @ K.T
    scores = torch.bmm(q_b, k_b.transpose(1, 2))

    # P = softmax(S / sqrt(D_HEAD))
    p = torch.nn.functional.softmax(scores / (D_HEAD**0.5), dim=-1)

    # O = P @ V
    output_b = torch.bmm(p, v_b)

    # Reshape back to original dimensions
    return output_b.view(B, N_CTX, H, D_HEAD)
