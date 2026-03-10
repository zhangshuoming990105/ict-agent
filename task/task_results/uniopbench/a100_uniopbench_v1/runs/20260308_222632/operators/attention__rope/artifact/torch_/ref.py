import torch

def torch_kernel(x: torch.Tensor) -> torch.Tensor:
    orig_shape = x.shape
    if x.dim() > 2:
        x = x.view(-1, x.shape[-1])
        
    seq_len, head_dim = x.shape
    theta = 10000.0
    
    x_ = x.float().reshape(seq_len, head_dim // 2, 2)
    x_c = torch.view_as_complex(x_)
    
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=x.device).float() / head_dim))
    
    t = torch.arange(seq_len, device=x.device).float()
    
    freqs_outer = torch.outer(t, freqs)
    
    freqs_cis = torch.polar(torch.ones_like(freqs_outer), freqs_outer)
    
    x_out_c = x_c * freqs_cis
    
    x_out = torch.view_as_real(x_out_c).flatten(1)
    return x_out.view(*orig_shape).type_as(x)