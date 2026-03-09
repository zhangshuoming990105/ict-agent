import torch


def torch_kernel(input_tensor, kernel_tensor):
    """PyTorch参考实现 for depthwiseconv - 只负责纯计算"""
    # 输入应该已经是PyTorch期望的格式：
    # input_tensor: (1, C, H, W)
    # kernel_tensor: (C, 1, kH, kW)

    # Depthwise convolution
    output = torch.nn.functional.conv2d(
        input_tensor,
        kernel_tensor,
        groups=input_tensor.shape[1],  # groups = channels for depthwise
        padding=0,
    )

    return output
