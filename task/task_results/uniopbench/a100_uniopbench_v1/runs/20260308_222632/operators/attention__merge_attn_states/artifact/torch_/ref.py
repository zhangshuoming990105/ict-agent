import torch

def torch_kernel(output_lse, prefix_output, prefix_lse, suffix_output, suffix_lse):
    p_lse = prefix_lse.unsqueeze(-1)
    s_lse = suffix_lse.unsqueeze(-1)
    
    p_lse = torch.where(torch.isinf(p_lse), torch.full_like(p_lse, float('-inf')), p_lse)
    s_lse = torch.where(torch.isinf(s_lse), torch.full_like(s_lse, float('-inf')), s_lse)
    
    max_lse = torch.maximum(p_lse, s_lse)
    p_se = torch.exp(p_lse - max_lse)
    s_se = torch.exp(s_lse - max_lse)
    out_se = p_se + s_se
    
    p_scale = p_se / out_se
    s_scale = s_se / out_se
    
    p_scale = p_scale.transpose(0, 1)
    s_scale = s_scale.transpose(0, 1)
    
    output = prefix_output * p_scale + suffix_output * s_scale
    
    if output_lse is not None:
        new_lse = torch.log(out_se.squeeze(-1)) + max_lse.squeeze(-1)
        output_lse.copy_(new_lse)
    
    return output