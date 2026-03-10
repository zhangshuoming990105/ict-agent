import torch


def torch_kernel(x: torch.Tensor, 
                 batch_size: int = None, 
                 channels: int = None, 
                 input_H: int = None, 
                 kernel_size: int = None, 
                 stride: int = None) -> torch.Tensor:
    """
    Average pooling 2D reference implementation.
    
    The new test framework (test.py) passes all scalar specs as arguments.
    The old framework (check_cuda.py) only passes kernel_size and stride.
    We handle both by making all scalar parameters optional and checking which are provided.
    """
    # Old framework passes kernel_size as batch_size and stride as channels
    # when it only passes 2 args
    if batch_size is not None and channels is not None and input_H is None:
        # Old framework: torch_kernel(x, kernel_size, stride)
        kernel_size = batch_size
        stride = channels
    elif kernel_size is None or stride is None:
        raise ValueError("kernel_size and stride must be provided")
    
    return torch.nn.functional.avg_pool2d(x, kernel_size=kernel_size, stride=stride)
