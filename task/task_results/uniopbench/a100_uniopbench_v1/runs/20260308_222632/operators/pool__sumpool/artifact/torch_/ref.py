import torch


def torch_kernel(x: torch.Tensor, batch_size: int, channels: int, input_H: int, kernel_size: int, stride: int) -> torch.Tensor:
    # The test framework passes all scalar_specs as arguments
    # We only need kernel_size and stride for the actual pooling operation
    # 
    # Input x is in NHWC format (batch, height, width, channels)
    # but avg_pool2d expects NCHW format (batch, channels, height, width)
    # Convert NHWC to NCHW
    x_nchw = x.permute(0, 3, 1, 2).contiguous()
    
    result = torch.nn.functional.avg_pool2d(x_nchw, kernel_size=kernel_size, stride=stride) * (
        kernel_size * kernel_size
    )
    return result
