import torch
import torch.nn.functional as F
import math


def build_mean_kernel(K, device):
    val = 1.0 / float(K * K)
    kernel = torch.full(
        (1, 1, K, K),
        val,
        dtype=torch.float32,
        device=device,
    )
    return kernel


def build_gaussian_kernel(K, device):
    sigma = 0.3 * ((K - 1) * 0.5 - 1.0) + 0.8
    kernel = torch.empty(K * K, dtype=torch.float32, device=device)
    cx = (K - 1) * 0.5
    sum_val = 0.0

    for idx in range(K * K):

        x = idx % K
        y = idx // K

        dx = x - cx
        dy = y - cx

        w = math.exp(-(dx * dx / (2 * sigma * sigma) +
                       dy * dy / (2 * sigma * sigma)))

        kernel[idx] = w
        sum_val += w

    kernel /= sum_val
    return kernel.view(1, 1, K, K)


def torch_kernel(
    src,
    temp_kernel,
    temp_sum,
    N,
    H,
    W,
    adaptive_method,
    threshold_type,
    block_size,
    max_value,
    c,
):

    device = src.device
    x = src.to(torch.float32).unsqueeze(1)
    pad = block_size // 2
    x = F.pad(x, (pad, pad, pad, pad), mode="replicate")

    if adaptive_method == 0:
        weight = build_mean_kernel(block_size, device)
    else:
        weight = build_gaussian_kernel(block_size, device)

    local_mean = F.conv2d(x, weight)

    if threshold_type == 0:
        idelta = math.ceil(c)
    else:
        idelta = math.floor(c)

    sval = src.to(torch.int32).unsqueeze(1) + idelta

    res_i = torch.floor(local_mean + 0.5).to(torch.int32)
    res_i = torch.clamp(res_i, 0, 255)


    if threshold_type == 0:
        cond = sval > res_i
    else:
        cond = sval <= res_i

    maxv = int(round(max_value))

    dst = torch.where(
        cond,
        torch.tensor(maxv, dtype=torch.uint8, device=device),
        torch.tensor(0, dtype=torch.uint8, device=device),
    )

    return dst.squeeze(1)