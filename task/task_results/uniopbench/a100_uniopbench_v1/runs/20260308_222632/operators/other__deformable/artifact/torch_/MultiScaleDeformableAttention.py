"""
Multi-Scale Deformable Attention reference implementation.
This is a PyTorch-based reference implementation for the deformable attention operator.
"""

import torch
import torch.nn.functional as F


def ms_deform_attn_forward(
    value,
    spatial_shapes,
    level_start_index,
    sampling_locations,
    attention_weights,
    im2col_step=1
):
    """
    Multi-scale deformable attention forward pass.
    
    Args:
        value: (N, S, M, D) - value features where S is total spatial size
        spatial_shapes: (L, 2) - spatial shapes for each level (H, W)
        level_start_index: (L,) - start index for each level in the flattened spatial dimension
        sampling_locations: (N, Lq, M, L, P, 2) - sampling locations (normalized 0-1)
        attention_weights: (N, Lq, M, L, P) - attention weights
        im2col_step: int - step size for im2col (not used in this implementation)
        
    Returns:
        output: (N, Lq, M*D) - aggregated features
    """
    N, S, M, D = value.shape
    _, Lq, _, L, P, _ = sampling_locations.shape
    
    # Initialize output
    output = torch.zeros(N, Lq, M, D, dtype=value.dtype, device=value.device)
    
    # Process each level
    for level_idx in range(L):
        # Get level parameters
        H, W = spatial_shapes[level_idx]
        level_start = level_start_index[level_idx]
        
        # Extract value for this level: (N, H*W, M, D)
        if level_idx < L - 1:
            level_end = level_start_index[level_idx + 1]
        else:
            level_end = S
        
        value_level = value[:, level_start:level_end, :, :]  # (N, H*W, M, D)
        value_level = value_level.reshape(N, H, W, M, D)  # (N, H, W, M, D)
        
        # Get sampling locations for this level: (N, Lq, M, P, 2)
        sampling_loc_level = sampling_locations[:, :, :, level_idx, :, :]
        
        # Get attention weights for this level: (N, Lq, M, P)
        attn_weight_level = attention_weights[:, :, :, level_idx, :]
        
        # Process each sample in the batch
        for n in range(N):
            for q in range(Lq):
                for m in range(M):
                    for p in range(P):
                        # Get normalized sampling location
                        loc_h = sampling_loc_level[n, q, m, p, 0]  # normalized [0, 1]
                        loc_w = sampling_loc_level[n, q, m, p, 1]  # normalized [0, 1]
                        
                        # Convert to absolute coordinates
                        h_abs = loc_h * (H - 1)
                        w_abs = loc_w * (W - 1)
                        
                        # Bilinear interpolation
                        h_low = torch.floor(h_abs).long()
                        w_low = torch.floor(w_abs).long()
                        h_high = h_low + 1
                        w_high = w_low + 1
                        
                        # Clamp to valid range
                        h_low = torch.clamp(h_low, 0, H - 1)
                        h_high = torch.clamp(h_high, 0, H - 1)
                        w_low = torch.clamp(w_low, 0, W - 1)
                        w_high = torch.clamp(w_high, 0, W - 1)
                        
                        # Compute interpolation weights
                        lh = h_abs - h_low.float()
                        lw = w_abs - w_low.float()
                        hh = 1 - lh
                        hw = 1 - lw
                        
                        # Boundary check
                        if h_abs < 0 or h_abs > H - 1 or w_abs < 0 or w_abs > W - 1:
                            continue
                        
                        # Get values at the four corners
                        v1 = value_level[n, h_low, w_low, m, :] * (hh * hw)
                        v2 = value_level[n, h_low, w_high, m, :] * (hh * lw)
                        v3 = value_level[n, h_high, w_low, m, :] * (lh * hw)
                        v4 = value_level[n, h_high, w_high, m, :] * (lh * lw)
                        
                        # Interpolated value
                        value_interp = v1 + v2 + v3 + v4
                        
                        # Apply attention weight and accumulate
                        attn_w = attn_weight_level[n, q, m, p]
                        output[n, q, m, :] += value_interp * attn_w
    
    # Reshape output to (N, Lq, M*D)
    output = output.reshape(N, Lq, M * D)
    
    return output
