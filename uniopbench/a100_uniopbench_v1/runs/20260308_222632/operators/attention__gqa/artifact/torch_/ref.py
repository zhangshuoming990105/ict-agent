import torch
import torch.nn.functional as F


def torch_kernel(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    Grouped Query Attention (GQA) implementation.
    Q: [batch, num_q_heads, M, D] or [batch*num_q_heads, M, D]
    K: [batch, num_kv_heads, N, D] or [batch*num_kv_heads, N, D]
    V: [batch, num_kv_heads, N, D] or [batch*num_kv_heads, N, D]

    Each group of Q heads shares one KV head.
    """
    # Handle both 4D and 3D inputs
    if Q.dim() == 4:
        # New framework: [batch, num_heads, seq, dim]
        batch, num_q_heads, M, D = Q.shape
        _, num_kv_heads, N, _ = K.shape
        
        # Reshape to 3D for processing
        Q = Q.reshape(batch * num_q_heads, M, D)
        K = K.reshape(batch * num_kv_heads, N, D)
        V = V.reshape(batch * num_kv_heads, N, D)
        
        # Process and reshape back to 4D
        result = _gqa_attention_3d(Q, K, V)
        return result.reshape(batch, num_q_heads, M, D)
    else:
        # Old framework: [batch*num_heads, seq, dim]
        return _gqa_attention_3d(Q, K, V)


def _gqa_attention_3d(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    Core GQA implementation for 3D tensors.
    Q: [batch*num_q_heads, M, D]
    K: [batch*num_kv_heads, N, D]
    V: [batch*num_kv_heads, N, D]
    """
    num_q_heads = Q.shape[0]
    num_kv_heads = K.shape[0]
    group_size = num_q_heads // num_kv_heads

    # Process each Q head
    outputs = []
    for q_idx in range(num_q_heads):
        # Determine which KV head this Q head uses
        kv_idx = q_idx // group_size

        # Get single Q, K, V for this head
        q = Q[q_idx : q_idx + 1]  # [1, M, D]
        k = K[kv_idx : kv_idx + 1]  # [1, N, D]
        v = V[kv_idx : kv_idx + 1]  # [1, N, D]

        # Compute attention: O = softmax(Q @ K^T / sqrt(d)) @ V
        q_f32 = q.to(torch.float32)
        k_f32 = k.to(torch.float32)
        v_f32 = v.to(torch.float32)

        d = q.shape[-1]
        scores = torch.matmul(q_f32, k_f32.transpose(-2, -1)) / (d**0.5)
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v_f32)

        outputs.append(out)

    # Concatenate all outputs: [num_q_heads, M, D]
    result = torch.cat(outputs, dim=0)
    return result.to(Q.dtype)
