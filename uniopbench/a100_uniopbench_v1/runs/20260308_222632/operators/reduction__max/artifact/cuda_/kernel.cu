#include <cuda_runtime.h>
#include <float.h>

// Warp reduction for max
__device__ __forceinline__ float warpReduceMax(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

// Block reduction for max using shared memory
__device__ __forceinline__ float blockReduceMax(float val) {
    __shared__ float shared[32]; // One per warp
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;

    // Warp-level reduction
    val = warpReduceMax(val);

    // Write warp result to shared memory
    if (lane == 0) {
        shared[wid] = val;
    }
    __syncthreads();

    // Final reduction by first warp
    if (threadIdx.x < 32) {
        val = (threadIdx.x < (blockDim.x + 31) / 32) ? shared[threadIdx.x] : -FLT_MAX;
        val = warpReduceMax(val);
    }

    return val;
}

// Kernel for reduction max along dimension 0
// input: [rows, inner], output: [inner]
__global__ void reduction_max_kernel(const float* input, float* output, int rows, int inner) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (col < inner) {
        float max_val = -FLT_MAX;
        
        // Each thread computes max for one column
        for (int row = 0; row < rows; ++row) {
            max_val = fmaxf(max_val, input[row * inner + col]);
        }
        
        output[col] = max_val;
    }
}

// Optimized kernel for large rows using block-level reduction
// Each column is handled by multiple threads in a block
__global__ void reduction_max_kernel_large(const float* input, float* output, int rows, int inner) {
    int col = blockIdx.x;
    
    if (col < inner) {
        float max_val = -FLT_MAX;
        
        // Grid-stride loop over rows
        for (int row = threadIdx.x; row < rows; row += blockDim.x) {
            max_val = fmaxf(max_val, input[row * inner + col]);
        }
        
        // Block-level reduction
        max_val = blockReduceMax(max_val);
        
        // First thread writes result
        if (threadIdx.x == 0) {
            output[col] = max_val;
        }
    }
}

extern "C" {

void cuda_kernel(float* input, float* output, int rows, int inner) {
    // Choose kernel based on problem size
    // If rows is large, use block-level reduction per column
    // Otherwise, use simple one-thread-per-column approach
    
    if (rows >= 128) {
        // Large rows: use block reduction
        int threads = 256;
        int blocks = inner;
        reduction_max_kernel_large<<<blocks, threads>>>(input, output, rows, inner);
    } else {
        // Small to medium rows: one thread per column
        int threads = 256;
        int blocks = (inner + threads - 1) / threads;
        reduction_max_kernel<<<blocks, threads>>>(input, output, rows, inner);
    }
}

} // extern "C"
