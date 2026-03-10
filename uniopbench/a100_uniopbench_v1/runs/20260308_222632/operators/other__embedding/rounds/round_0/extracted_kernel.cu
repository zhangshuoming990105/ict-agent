#include <cuda_runtime.h>
#include <cuda_fp16.h>

// Scalar version for float32
__global__ void embedding_kernel_fp32_scalar(
    const int* __restrict__ indices,
    const float* __restrict__ weight,
    float* __restrict__ output,
    int N,
    int emb_size,
    int max_vocab_size
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = N * emb_size;
    
    for (int idx = tid; idx < total_elements; idx += blockDim.x * gridDim.x) {
        int row = idx / emb_size;
        int col = idx % emb_size;
        
        int vocab_idx = indices[row];
        output[idx] = weight[vocab_idx * emb_size + col];
    }
}

// Vectorized version for float32 (using float4)
__global__ void embedding_kernel_fp32_vec4(
    const int* __restrict__ indices,
    const float* __restrict__ weight,
    float* __restrict__ output,
    int N,
    int emb_size,
    int max_vocab_size
) {
    int row = blockIdx.x;
    int col = threadIdx.x * 4;
    
    if (row < N && col < emb_size) {
        int vocab_idx = indices[row];
        int weight_offset = vocab_idx * emb_size + col;
        int output_offset = row * emb_size + col;
        
        if (col + 3 < emb_size) {
            float4 val = *reinterpret_cast<const float4*>(&weight[weight_offset]);
            *reinterpret_cast<float4*>(&output[output_offset]) = val;
        } else {
            // Handle remainder
            for (int i = 0; i < 4 && col + i < emb_size; i++) {
                output[output_offset + i] = weight[weight_offset + i];
            }
        }
    }
}

// Packed version for float32 (same as vec4 but optimized grid)
__global__ void embedding_kernel_fp32_packed(
    const int* __restrict__ indices,
    const float* __restrict__ weight,
    float* __restrict__ output,
    int N,
    int emb_size,
    int max_vocab_size
) {
    int row = blockIdx.x;
    int col = threadIdx.x * 4;
    
    if (row >= N) return;
    
    int vocab_idx = indices[row];
    const float* weight_row = weight + vocab_idx * emb_size;
    float* output_row = output + row * emb_size;
    
    for (int c = col; c < emb_size; c += blockDim.x * 4) {
        if (c + 3 < emb_size) {
            float4 val = *reinterpret_cast<const float4*>(&weight_row[c]);
            *reinterpret_cast<float4*>(&output_row[c]) = val;
        } else {
            for (int i = 0; i < 4 && c + i < emb_size; i++) {
                output_row[c + i] = weight_row[c + i];
            }
        }
    }
}

// Scalar version for float16
__global__ void embedding_kernel_fp16_scalar(
    const int* __restrict__ indices,
    const __half* __restrict__ weight,
    __half* __restrict__ output,
    int N,
    int emb_size,
    int max_vocab_size
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = N * emb_size;
    
    for (int idx = tid; idx < total_elements; idx += blockDim.x * gridDim.x) {
        int row = idx / emb_size;
        int col = idx % emb_size;
        
        int vocab_idx = indices[row];
        output[idx] = weight[vocab_idx * emb_size + col];
    }
}

// Vectorized version for float16 (using float4 = 8 halfs)
__global__ void embedding_kernel_fp16_vec4(
    const int* __restrict__ indices,
    const __half* __restrict__ weight,
    __half* __restrict__ output,
    int N,
    int emb_size,
    int max_vocab_size
) {
    int row = blockIdx.x;
    int col = threadIdx.x * 8;
    
    if (row < N && col < emb_size) {
        int vocab_idx = indices[row];
        int weight_offset = vocab_idx * emb_size + col;
        int output_offset = row * emb_size + col;
        
        if (col + 7 < emb_size) {
            float4 val = *reinterpret_cast<const float4*>(&weight[weight_offset]);
            *reinterpret_cast<float4*>(&output[output_offset]) = val;
        } else {
            // Handle remainder
            for (int i = 0; i < 8 && col + i < emb_size; i++) {
                output[output_offset + i] = weight[weight_offset + i];
            }
        }
    }
}

// Packed version for float16 (optimized grid)
__global__ void embedding_kernel_fp16_packed(
    const int* __restrict__ indices,
    const __half* __restrict__ weight,
    __half* __restrict__ output,
    int N,
    int emb_size,
    int max_vocab_size
) {
    int row = blockIdx.x;
    int col = threadIdx.x * 8;
    
    if (row >= N) return;
    
    int vocab_idx = indices[row];
    const __half* weight_row = weight + vocab_idx * emb_size;
    __half* output_row = output + row * emb_size;
    
    for (int c = col; c < emb_size; c += blockDim.x * 8) {
        if (c + 7 < emb_size) {
            float4 val = *reinterpret_cast<const float4*>(&weight_row[c]);
            *reinterpret_cast<float4*>(&output_row[c]) = val;
        } else {
            for (int i = 0; i < 8 && c + i < emb_size; i++) {
                output_row[c + i] = weight_row[c + i];
            }
        }
    }
}

extern "C" {

void cuda_kernel(
    void* indices,
    void* weight,
    void* output,
    int N,
    int emb_size,
    int op_type
) {
    const int* idx_ptr = static_cast<const int*>(indices);
    
    // op_type: 0-2 for fp32 (scalar, vec, packed), 3-5 for fp16
    if (op_type < 3) {
        // float32
        const float* weight_ptr = static_cast<const float*>(weight);
        float* output_ptr = static_cast<float*>(output);
        
        if (op_type == 0) {
            // Scalar
            int block_size = 256;
            int grid_size = (N * emb_size + block_size - 1) / block_size;
            grid_size = min(grid_size, 1024);
            embedding_kernel_fp32_scalar<<<grid_size, block_size>>>(
                idx_ptr, weight_ptr, output_ptr, N, emb_size, 0
            );
        } else if (op_type == 1) {
            // Vectorized (float4)
            int threads = (emb_size + 3) / 4;
            threads = min(threads, 256);
            embedding_kernel_fp32_vec4<<<N, threads>>>(
                idx_ptr, weight_ptr, output_ptr, N, emb_size, 0
            );
        } else {
            // Packed (float4 with grid-stride)
            int threads = min((emb_size + 3) / 4, 256);
            embedding_kernel_fp32_packed<<<N, threads>>>(
                idx_ptr, weight_ptr, output_ptr, N, emb_size, 0
            );
        }
    } else {
        // float16
        const __half* weight_ptr = static_cast<const __half*>(weight);
        __half* output_ptr = static_cast<__half*>(output);
        
        if (op_type == 3) {
            // Scalar
            int block_size = 256;
            int grid_size = (N * emb_size + block_size - 1) / block_size;
            grid_size = min(grid_size, 1024);
            embedding_kernel_fp16_scalar<<<grid_size, block_size>>>(
                idx_ptr, weight_ptr, output_ptr, N, emb_size, 0
            );
        } else if (op_type == 4) {
            // Vectorized (8 halfs = float4)
            int threads = (emb_size + 7) / 8;
            threads = min(threads, 256);
            embedding_kernel_fp16_vec4<<<N, threads>>>(
                idx_ptr, weight_ptr, output_ptr, N, emb_size, 0
            );
        } else {
            // Packed (8 halfs with grid-stride)
            int threads = min((emb_size + 7) / 8, 256);
            embedding_kernel_fp16_packed<<<N, threads>>>(
                idx_ptr, weight_ptr, output_ptr, N, emb_size, 0
            );
        }
    }
}

} // extern "C"
