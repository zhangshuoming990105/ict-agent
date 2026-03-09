import torch
from get_data import Params, get_cuda_torch_inputs

params = Params()
cuda_all_inputs, torch_all_inputs, outputs = get_cuda_torch_inputs(params)

print(f"cuda_all_inputs: {len(cuda_all_inputs)} items")
for i, inp in enumerate(cuda_all_inputs):
    if isinstance(inp, torch.Tensor):
        print(f"  [{i}] Tensor: shape={inp.shape}, dtype={inp.dtype}, device={inp.device}")
    else:
        print(f"  [{i}] Scalar: {inp}")

print(f"\ntorch_all_inputs: {len(torch_all_inputs)} items")
for i, inp in enumerate(torch_all_inputs):
    if isinstance(inp, torch.Tensor):
        print(f"  [{i}] Tensor: shape={inp.shape}, dtype={inp.dtype}, device={inp.device}")
        if inp.numel() <= 10:
            print(f"       Values: {inp}")
    else:
        print(f"  [{i}] Scalar: {inp}")

print(f"\noutputs: {len(outputs)} items")
for i, out in enumerate(outputs):
    if isinstance(out, torch.Tensor):
        print(f"  [{i}] Tensor: shape={out.shape}, dtype={out.dtype}, device={out.device}")
