#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>
#include <cstdio>

#define WARP_SIZE 32

// Optimized MHA kernel with better memory access patterns and parallelization
__global__ void mha_kernel(
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    float* __restrict__ output,
    int batch_size,
    int seq_len,
    int num_heads,
    int head_dim
) {
    extern __shared__ float shared_mem[];
    
    // Each block processes one (batch, seq) pair
    int bs_idx = blockIdx.x;
    if (bs_idx >= batch_size * seq_len) return;
    
    int b = bs_idx / seq_len;
    int n = bs_idx % seq_len;
    
    // Base offset: (b, n, :, :) => (H, D_HEAD)
    int base_offset = (b * seq_len + n) * num_heads * head_dim;
    const float* q_base = q + base_offset;
    const float* k_base = k + base_offset;
    const float* v_base = v + base_offset;
    float* out_base = output + base_offset;
    
    float scale = rsqrtf((float)head_dim);
    
    // Shared memory: scores (H*H)
    float* scores = shared_mem;
    
    // Step 1: Compute attention scores (H, H) in parallel
    // Each thread computes multiple elements if needed
    int total_score_elements = num_heads * num_heads;
    
    for (int idx = threadIdx.x; idx < total_score_elements; idx += blockDim.x) {
        int i = idx / num_heads;  // query head
        int j = idx % num_heads;  // key head
        
        float sum = 0.0f;
        
        // Vectorized dot product when possible
        const float* q_ptr = q_base + i * head_dim;
        const float* k_ptr = k_base + j * head_dim;
        
        // Unroll for better ILP
        int d = 0;
        for (; d + 3 < head_dim; d += 4) {
            sum += q_ptr[d] * k_ptr[d];
            sum += q_ptr[d+1] * k_ptr[d+1];
            sum += q_ptr[d+2] * k_ptr[d+2];
            sum += q_ptr[d+3] * k_ptr[d+3];
        }
        for (; d < head_dim; d++) {
            sum += q_ptr[d] * k_ptr[d];
        }
        
        scores[idx] = sum * scale;
    }
    __syncthreads();
    
    // Step 2: Softmax over each row
    // Assign threads to rows for better parallelism
    for (int i = threadIdx.x; i < num_heads; i += blockDim.x) {
        float* row = scores + i * num_heads;
        
        // Find max
        float max_val = row[0];
        for (int j = 1; j < num_heads; j++) {
            max_val = fmaxf(max_val, row[j]);
        }
        
        // Compute exp and sum in one pass
        float sum_exp = 0.0f;
        for (int j = 0; j < num_heads; j++) {
            float val = expf(row[j] - max_val);
            row[j] = val;
            sum_exp += val;
        }
        
        // Normalize
        float inv_sum = 1.0f / sum_exp;
        for (int j = 0; j < num_heads; j++) {
            row[j] *= inv_sum;
        }
    }
    __syncthreads();
    
    // Step 3: Compute output = P @ V
    // Each thread computes one or more output elements
    int total_out_elements = num_heads * head_dim;
    
    for (int idx = threadIdx.x; idx < total_out_elements; idx += blockDim.x) {
        int i = idx / head_dim;  // output head
        int d = idx % head_dim;  // dimension
        
        float sum = 0.0f;
        const float* prob_row = scores + i * num_heads;
        
        // Accumulate: sum over all heads
        for (int j = 0; j < num_heads; j++) {
            sum += prob_row[j] * v_base[j * head_dim + d];
        }
        
        out_base[idx] = sum;
    }
}

extern "C" {

void cuda_kernel(
    const float* q,
    const float* k,
    const float* v,
    float* output,
    int batch_size,
    int seq_len,
    int num_heads,
    int head_dim
) {
    // Each block handles one (batch, seq) pair
    int total_blocks = batch_size * seq_len;
    
    // Shared memory: scores (H*H)
    int shared_mem_size = num_heads * num_heads * sizeof(float);
    
    // Use 256 threads per block for good occupancy on A100
    int threads_per_block = 256;
    
    mha_kernel<<<total_blocks, threads_per_block, shared_mem_size>>>(
        q, k, v, output,
        batch_size, seq_len, num_heads, head_dim
    );
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA kernel launch error: %s\n", cudaGetErrorString(err));
    }
}

} // extern "C"
