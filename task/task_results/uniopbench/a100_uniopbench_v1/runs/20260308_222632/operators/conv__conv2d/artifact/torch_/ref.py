import torch


def torch_kernel(input_tensor, kernel_tensor, batch_size, in_height, in_width, 
                 in_channels, out_channels, kernel_height, kernel_width, 
                 stride, padding):
    """PyTorch reference implementation for conv2d
    
    Args:
        input_tensor: Input tensor in NCHW format (batch, in_channels, in_height, in_width)
        kernel_tensor: Kernel tensor (out_channels, in_channels, kernel_height, kernel_width)
        batch_size: Batch size (unused, for interface compatibility)
        in_height: Input height (unused, for interface compatibility)
        in_width: Input width (unused, for interface compatibility)
        in_channels: Input channels (unused, for interface compatibility)
        out_channels: Output channels (unused, for interface compatibility)
        kernel_height: Kernel height (unused, for interface compatibility)
        kernel_width: Kernel width (unused, for interface compatibility)
        stride: Convolution stride
        padding: Convolution padding
    
    Returns:
        Output tensor from conv2d operation
    """
    output = torch.nn.functional.conv2d(
        input_tensor, kernel_tensor, stride=stride, padding=padding
    )
    return output
