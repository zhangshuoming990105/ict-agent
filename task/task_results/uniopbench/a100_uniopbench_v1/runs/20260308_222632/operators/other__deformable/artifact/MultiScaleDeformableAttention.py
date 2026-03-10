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
    
    # Ensure correct dtypes
    if spatial_shapes.dtype != torch.int64:
        spatial_shapes = spatial_shapes.to(torch.int64)
    if level_start_index.dtype != torch.int64:
        level_start_index = level_start_index.to(torch.int64)
    
    # Initialize output
    output = torch.zeros(N, Lq, M, D, dtype=value.dtype, device=value.device)
    
    # Process each level
    for level_idx in range(L):
        # Get level parameters
        H = int(spatial_shapes[level_idx, 0].item())
        W = int(spatial_shapes[level_idx, 1].item())
        level_start = int(level_start_index[level_idx].item())
        
        # Extract value for this level: (N, H*W, M, D)
        if level_idx < L - 1:
            level_end = int(level_start_index[level_idx + 1].item())
        else:
            level_end = S
        
        level_size = level_end - level_start
        
        # Verify the level size matches H*W
        if level_size != H * W:
            raise ValueError(f"Level {level_idx}: size mismatch. Expected {H}*{W}={H*W}, got {level_size}")
        
        value_level = value[:, level_start:level_end, :, :]  # (N, H*W, M, D)
        value_level = value_level.reshape(N, H, W, M, D)  # (N, H, W, M, D)
        
        # Get sampling locations for this level: (N, Lq, M, P, 2)
        sampling_loc_level = sampling_locations[:, :, :, level_idx, :, :]
        
        # Get attention weights for this level: (N, Lq, M, P)
        attn_weight_level = attention_weights[:, :, :, level_idx, :]
        
        # Vectorized interpolation for this level
        # sampling_loc_level: (N, Lq, M, P, 2)
        loc_h = sampling_loc_level[..., 0]  # (N, Lq, M, P)
        loc_w = sampling_loc_level[..., 1]  # (N, Lq, M, P)
        
        # Convert to absolute coordinates
        h_abs = loc_h * (H - 1)  # (N, Lq, M, P)
        w_abs = loc_w * (W - 1)  # (N, Lq, M, P)
        
        # Get integer parts
        h_low = torch.floor(h_abs).long()  # (N, Lq, M, P)
        w_low = torch.floor(w_abs).long()  # (N, Lq, M, P)
        h_high = h_low + 1  # (N, Lq, M, P)
        w_high = w_low + 1  # (N, Lq, M, P)
        
        # Clamp to valid range
        h_low = torch.clamp(h_low, 0, H - 1)
        h_high = torch.clamp(h_high, 0, H - 1)
        w_low = torch.clamp(w_low, 0, W - 1)
        w_high = torch.clamp(w_high, 0, W - 1)
        
        # Compute interpolation weights
        lh = h_abs - h_low.float()  # (N, Lq, M, P)
        lw = w_abs - w_low.float()  # (N, Lq, M, P)
        hh = 1 - lh  # (N, Lq, M, P)
        hw = 1 - lw  # (N, Lq, M, P)
        
        # Boundary mask
        valid_mask = (h_abs >= 0) & (h_abs <= H - 1) & (w_abs >= 0) & (w_abs <= W - 1)  # (N, Lq, M, P)
        
        # Get values at the four corners
        # value_level: (N, H, W, M, D)
        # We need to gather values for each (N, Lq, M, P) combination
        
        # Reshape for gathering
        # value_level: (N, H, W, M, D) -> (N, M, H, W, D)
        value_level_perm = value_level.permute(0, 3, 1, 2, 4)  # (N, M, H, W, D)
        
        # Flatten spatial dimensions to use advanced indexing
        # value_level_perm: (N, M, H*W, D)
        value_level_flat = value_level_perm.reshape(N, M, H*W, D)
        
        # Compute flat indices
        idx_ll = (h_low * W + w_low).unsqueeze(-1).expand(-1, -1, -1, -1, D)  # (N, Lq, M, P, D)
        idx_lh = (h_low * W + w_high).unsqueeze(-1).expand(-1, -1, -1, -1, D)  # (N, Lq, M, P, D)
        idx_hl = (h_high * W + w_low).unsqueeze(-1).expand(-1, -1, -1, -1, D)  # (N, Lq, M, P, D)
        idx_hh = (h_high * W + w_high).unsqueeze(-1).expand(-1, -1, -1, -1, D)  # (N, Lq, M, P, D)
        
        # Expand value_level_flat for gathering
        # value_level_flat: (N, M, H*W, D) -> (N, Lq, M, H*W, D)
        value_level_flat = value_level_flat.unsqueeze(1).expand(N, Lq, M, H*W, D)
        
        # Gather values
        v_ll = torch.gather(value_level_flat, 3, idx_ll)  # (N, Lq, M, P, D)
        v_lh = torch.gather(value_level_flat, 3, idx_lh)  # (N, Lq, M, P, D)
        v_hl = torch.gather(value_level_flat, 3, idx_hl)  # (N, Lq, M, P, D)
        v_hh = torch.gather(value_level_flat, 3, idx_hh)  # (N, Lq, M, P, D)
        
        # Compute interpolated values
        w_ll = (hh * hw).unsqueeze(-1)  # (N, Lq, M, P, 1)
        w_lh = (hh * lw).unsqueeze(-1)  # (N, Lq, M, P, 1)
        w_hl = (lh * hw).unsqueeze(-1)  # (N, Lq, M, P, 1)
        w_hh = (lh * lw).unsqueeze(-1)  # (N, Lq, M, P, 1)
        
        value_interp = w_ll * v_ll + w_lh * v_lh + w_hl * v_hl + w_hh * v_hh  # (N, Lq, M, P, D)
        
        # Apply valid mask
        value_interp = value_interp * valid_mask.unsqueeze(-1).float()  # (N, Lq, M, P, D)
        
        # Apply attention weights and sum over points
        attn_weight_level_exp = attn_weight_level.unsqueeze(-1)  # (N, Lq, M, P, 1)
        weighted_value = value_interp * attn_weight_level_exp  # (N, Lq, M, P, D)
        
        # Sum over points
        output += weighted_value.sum(dim=3)  # (N, Lq, M, D)
    
    # Reshape output to (N, Lq, M*D)
    output = output.reshape(N, Lq, M * D)
    
    return output
