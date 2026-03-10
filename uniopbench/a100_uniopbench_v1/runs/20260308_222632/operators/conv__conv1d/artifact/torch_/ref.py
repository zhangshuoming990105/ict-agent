import torch


def torch_kernel(input_tensor, kernel_tensor):
    """简化的PyTorch参考实现 for conv1d"""
    # 使用1D卷积实现，添加必要的维度
    input_expanded = input_tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, input_size)
    kernel_expanded = kernel_tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, kernel_size)

    result = torch.nn.functional.conv1d(input_expanded, kernel_expanded, padding=0)
    return result.squeeze()  # 移除额外的维度
