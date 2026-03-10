#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cmath>

// Constants for performance tuning
#define WARP_SIZE 32
#define MAX_THREADS 256

// Warp-level reduction for sum
__inline__ __device__ float warpReduceSum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Warp-level reduction for max
__inline__ __device__ float warpReduceMax(float val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

// Kernel: Each block processes one query row for one Q head
// Grid: (batch * num_q_heads * M, 1, 1)
// Block: (min(N, MAX_THREADS), 1, 1)
__global__ void gqa_attention_kernel_4d(
    const half* __restrict__ Q,     // [batch, num_q_heads, M, K_dim]
    const half* __restrict__ K,     // [batch, num_kv_heads, N, K_dim]
    const half* __restrict__ V,     // [batch, num_kv_heads, N, K_dim]
    half* __restrict__ O,            // [batch, num_q_heads, M, K_dim]
    int batch,
    int num_q_heads,
    int num_kv_heads,
    int M,
    int K_dim,
    int N
) {
    int group_size = num_q_heads / num_kv_heads;
    
    // Determine which batch, Q head, and M row we're processing
    int global_idx = blockIdx.x;
    int total_heads = batch * num_q_heads;
    int total_rows = total_heads * M;
    
    if (global_idx >= total_rows) return;
    
    int b = global_idx / (num_q_heads * M);      // batch index
    int remaining = global_idx % (num_q_heads * M);
    int q_head = remaining / M;                   // Q head within batch
    int m_idx = remaining % M;                    // Row within M
    
    // Determine corresponding KV head
    int kv_head = q_head / group_size;
    
    // Thread and warp indices
    int tid = threadIdx.x;
    int lane_id = tid % WARP_SIZE;
    int warp_id = tid / WARP_SIZE;
    int num_warps = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;
    
    // Shared memory for warp-level reductions
    __shared__ float warp_max[MAX_THREADS / WARP_SIZE + 1];
    __shared__ float warp_sum[MAX_THREADS / WARP_SIZE + 1];
    __shared__ float warp_out[MAX_THREADS / WARP_SIZE + 1];
    
    float inv_sqrt_d = rsqrtf((float)K_dim);
    
    // Compute base offsets for 4D indexing
    // Q[b, q_head, m_idx, :] -> Q[b * num_q_heads * M * K_dim + q_head * M * K_dim + m_idx * K_dim]
    int q_base = b * num_q_heads * M * K_dim + q_head * M * K_dim + m_idx * K_dim;
    // K[b, kv_head, n, :] -> K[b * num_kv_heads * N * K_dim + kv_head * N * K_dim + n * K_dim]
    int k_base = b * num_kv_heads * N * K_dim + kv_head * N * K_dim;
    // V[b, kv_head, n, :] -> V[b * num_kv_heads * N * K_dim + kv_head * N * K_dim + n * K_dim]
    int v_base = b * num_kv_heads * N * K_dim + kv_head * N * K_dim;
    // O[b, q_head, m_idx, :] -> O[b * num_q_heads * M * K_dim + q_head * M * K_dim + m_idx * K_dim]
    int o_base = b * num_q_heads * M * K_dim + q_head * M * K_dim + m_idx * K_dim;
    
    // Step 1: Compute Q @ K^T scores and find max
    float thread_max = -INFINITY;
    for (int n = tid; n < N; n += blockDim.x) {
        float score = 0.0f;
        
        // Dot product: Q[b, q_head, m_idx, :] · K[b, kv_head, n, :]
        for (int k = 0; k < K_dim; k++) {
            float q_val = __half2float(Q[q_base + k]);
            float k_val = __half2float(K[k_base + n * K_dim + k]);
            score += q_val * k_val;
        }
        
        score *= inv_sqrt_d;
        thread_max = fmaxf(thread_max, score);
    }
    
    // Warp-level reduction for max
    float warp_max_val = warpReduceMax(thread_max);
    if (lane_id == 0) {
        warp_max[warp_id] = warp_max_val;
    }
    __syncthreads();
    
    // Final reduction across warps
    float global_max = -INFINITY;
    if (tid < num_warps) {
        global_max = warp_max[tid];
    }
    if (tid < WARP_SIZE) {
        global_max = warpReduceMax(global_max);
    }
    // Broadcast to all threads
    if (tid == 0) {
        warp_max[0] = global_max;
    }
    __syncthreads();
    global_max = warp_max[0];
    
    // Step 2: Compute exp(score - max) and sum
    float thread_sum = 0.0f;
    for (int n = tid; n < N; n += blockDim.x) {
        float score = 0.0f;
        
        for (int k = 0; k < K_dim; k++) {
            float q_val = __half2float(Q[q_base + k]);
            float k_val = __half2float(K[k_base + n * K_dim + k]);
            score += q_val * k_val;
        }
        
        score *= inv_sqrt_d;
        thread_sum += expf(score - global_max);
    }
    
    // Warp-level reduction for sum
    float warp_sum_val = warpReduceSum(thread_sum);
    if (lane_id == 0) {
        warp_sum[warp_id] = warp_sum_val;
    }
    __syncthreads();
    
    // Final reduction across warps
    float global_sum = 0.0f;
    if (tid < num_warps) {
        global_sum = warp_sum[tid];
    }
    if (tid < WARP_SIZE) {
        global_sum = warpReduceSum(global_sum);
    }
    if (tid == 0) {
        warp_sum[0] = global_sum;
    }
    __syncthreads();
    global_sum = warp_sum[0];
    
    float inv_sum = 1.0f / global_sum;
    
    // Step 3: Compute output = softmax @ V
    for (int k = 0; k < K_dim; k++) {
        float thread_out = 0.0f;
        
        for (int n = tid; n < N; n += blockDim.x) {
            // Recompute attention weight
            float score = 0.0f;
            for (int kk = 0; kk < K_dim; kk++) {
                float q_val = __half2float(Q[q_base + kk]);
                float k_val = __half2float(K[k_base + n * K_dim + kk]);
                score += q_val * k_val;
            }
            score *= inv_sqrt_d;
            float attn_weight = expf(score - global_max) * inv_sum;
            
            // Multiply with V
            float v_val = __half2float(V[v_base + n * K_dim + k]);
            thread_out += attn_weight * v_val;
        }
        
        // Warp-level reduction for output
        float warp_out_val = warpReduceSum(thread_out);
        if (lane_id == 0) {
            warp_out[warp_id] = warp_out_val;
        }
        __syncthreads();
        
        // Final reduction across warps
        float final_out = 0.0f;
        if (tid < num_warps) {
            final_out = warp_out[tid];
        }
        if (tid < WARP_SIZE) {
            final_out = warpReduceSum(final_out);
        }
        
        // Write output
        if (tid == 0) {
            O[o_base + k] = __float2half(final_out);
        }
        __syncthreads();
    }
}

extern "C" {

void cuda_kernel(
    half* Q,
    half* K,
    half* V,
    half* O,
    int batch,
    int num_q_heads,
    int num_kv_heads,
    int M,
    int K_dim,
    int N
) {
    // Launch configuration
    // Each block processes one (batch, q_head, m) tuple
    int num_blocks = batch * num_q_heads * M;
    int num_threads = min(N, MAX_THREADS);
    
    gqa_attention_kernel_4d<<<num_blocks, num_threads>>>(
        Q, K, V, O,
        batch, num_q_heads, num_kv_heads,
        M, K_dim, N
    );
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        // Handle error (in production, proper error handling should be done)
    }
}

} // extern "C"
