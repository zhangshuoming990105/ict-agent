/*
 * CUDA kernel for Causal Softmax operator
 *
 * Causal Softmax applies softmax with a causal mask:
 * For each position i in a sequence, normalize over positions [0, i].
 * output[i] = exp(x[i] - max) / sum(exp(x[j] - max) for j in [0, i])
 *
 * This implementation:
 * - Assigns one warp (32 threads) per sequence for coalesced memory access
 * - Uses warp-level reductions for max and sum computation
 * - Processes sequences in a grid-stride loop for flexibility
 */

#include <cuda_runtime.h>
#include <math.h>

// Warp size constant
#define WARP_SIZE 32

// Warp-level reduction helpers using __shfl_down_sync
__device__ __forceinline__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

/*
 * Causal Softmax CUDA kernel
 *
 * Each warp processes one sequence row. For position i, we compute:
 * 1. max_val = max(x[0], ..., x[i])
 * 2. exp_sum = sum(exp(x[j] - max_val) for j in [0, i])
 * 3. output[i] = exp(x[i] - max_val) / exp_sum
 *
 * Grid: (num_warps_needed, 1, 1) where num_warps_needed = ceil(batch_size / warps_per_block)
 * Block: (WARP_SIZE, warps_per_block, 1)
 */
__global__ void causal_softmax_kernel_float(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int batch_size,
    const int seq_len
) {
    // One warp per sequence
    const int warp_id = blockIdx.x * blockDim.y + threadIdx.y;
    const int lane_id = threadIdx.x;
    
    if (warp_id >= batch_size) return;
    
    const float* seq_input = input + warp_id * seq_len;
    float* seq_output = output + warp_id * seq_len;
    
    // Process each position i in the sequence
    for (int i = 0; i < seq_len; i++) {
        // Step 1: Compute max over [0, i]
        float local_max = -INFINITY;
        for (int j = lane_id; j <= i; j += WARP_SIZE) {
            local_max = fmaxf(local_max, seq_input[j]);
        }
        float max_val = warp_reduce_max(local_max);
        max_val = __shfl_sync(0xffffffff, max_val, 0);  // Broadcast from lane 0
        
        // Step 2: Compute sum(exp(x[j] - max_val)) for j in [0, i]
        float local_sum = 0.0f;
        for (int j = lane_id; j <= i; j += WARP_SIZE) {
            local_sum += expf(seq_input[j] - max_val);
        }
        float exp_sum = warp_reduce_sum(local_sum);
        exp_sum = __shfl_sync(0xffffffff, exp_sum, 0);  // Broadcast from lane 0
        
        // Step 3: Compute output[i] = exp(x[i] - max_val) / exp_sum
        if (lane_id == 0) {
            float exp_val = expf(seq_input[i] - max_val);
            seq_output[i] = exp_val / exp_sum;
        }
    }
}

/*
 * Host function to launch the causal softmax kernel
 *
 * This is the C interface expected by the Python wrapper.
 * The function must be named "cuda_kernel" to match the framework's expectations.
 */
extern "C" void cuda_kernel(
    const float* input,
    float* output,
    int batch_size,
    int seq_len
) {
    // Configure kernel launch parameters
    // Use multiple warps per block for better occupancy
    const int warps_per_block = 4;  // Tune this for best performance
    const int num_blocks = (batch_size + warps_per_block - 1) / warps_per_block;
    
    dim3 block_dim(WARP_SIZE, warps_per_block, 1);
    dim3 grid_dim(num_blocks, 1, 1);
    
    // Launch kernel
    causal_softmax_kernel_float<<<grid_dim, block_dim>>>(
        input,
        output,
        batch_size,
        seq_len
    );
    
    // Synchronize to catch any errors
    cudaDeviceSynchronize();
}
