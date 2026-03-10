"""PyTorch reference implementation for Causal Softmax operator."""

import torch


def torch_kernel(x: torch.Tensor) -> torch.Tensor:
    """Causal Softmax: Apply causal mask then softmax.
    
    Causal mask ensures that position i can only see positions <= i.
    This is typically used in autoregressive models like transformers.
    
    Args:
        x: Input tensor of shape (batch, seq_len) or (batch, num_heads, seq_len)
        
    Returns:
        Output tensor with causal softmax applied along the last dimension
    """
    # Convert to float32 for stable computation
    dtype = x.dtype
    x_float = x.to(torch.float32)
    
    # Create causal mask: lower triangular matrix
    # For each row i, positions > i should be masked (set to -inf)
    seq_len = x_float.shape[-1]
    
    # Causal Softmax implementation matching CUDA kernel logic:
    # For each position i in a sequence, normalize over valid positions [0, i+1).
    # This means for output[i], the denominator is sum(exp(x[0]...x[i])).
    # This is different from standard causal attention where we have a query i attending to keys j<=i.
    # Here, we treat the input as a single sequence of values.
    #
    # Logic:
    # 1. Compute exp(x - max)
    # 2. Compute cumulative sum of exp values (cumsum)
    # 3. Divide exp values by their corresponding cumsum value
    
    # Subtract max for stability
    x_max = x_float.max(dim=-1, keepdim=True)[0]
    exp_x = torch.exp(x_float - x_max)
    
    # Compute cumulative sum along the last dimension
    # cumsum[i] = sum(exp_x[0]...exp_x[i])
    cumsum_exp = torch.cumsum(exp_x, dim=-1)
    
    # Normalize
    # result[i] = exp_x[i] / cumsum_exp[i]
    result = exp_x / cumsum_exp
    
    # Convert back to original dtype
    return result.to(dtype)
