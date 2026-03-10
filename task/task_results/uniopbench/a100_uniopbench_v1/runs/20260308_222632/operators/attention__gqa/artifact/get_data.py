import torch
import ctypes
from dataclasses import dataclass
from optest.tools.checker import SEED


@dataclass
class Params:
    # GQA configuration: 16 Q heads share 2 KV heads (group_size=8)
    batch: int = 1
    num_q_heads: int = 16
    num_kv_heads: int = 2
    M: int = 16  # SEQ_Q
    K_dim: int = 16  # HEAD_DIM
    N: int = 512  # SEQ_KV


def get_cuda_argtypes():
    # half* Q, half* K, half* V, half* O, int batch, int num_q_heads, int num_kv_heads, int M, int K_dim, int N
    return [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
    ]


def get_cuda_torch_inputs(params: Params):
    torch.manual_seed(SEED)

    device = "cuda"
    dtype = torch.float16

    # GQA Layout:
    # Q: [batch * num_q_heads, M, K_dim]  -> [16, 16, 16]
    # K: [batch * num_kv_heads, N, K_dim] -> [2, 512, 16]
    # V: [batch * num_kv_heads, N, K_dim] -> [2, 512, 16]
    bq = params.batch * params.num_q_heads
    bkv = params.batch * params.num_kv_heads

    Q = torch.randn(bq, params.M, params.K_dim, device=device, dtype=dtype).normal_(
        mean=0.0, std=0.5
    )
    K = torch.randn(bkv, params.N, params.K_dim, device=device, dtype=dtype).normal_(
        mean=0.0, std=0.5
    )
    V = torch.randn(bkv, params.N, params.K_dim, device=device, dtype=dtype).normal_(
        mean=0.0, std=0.5
    )

    O = torch.empty_like(Q)

    cuda_all_inputs = [
        Q,
        K,
        V,
        O,
        params.batch,
        params.num_q_heads,
        params.num_kv_heads,
        params.M,
        params.K_dim,
        params.N,
    ]
    torch_all_inputs = [Q, K, V]
    cuda_output_tensors = [O]
    return cuda_all_inputs, torch_all_inputs, cuda_output_tensors


def cuda_output_tensor_transform(t):
    return t
