#include <cuda_runtime.h>
#include <cuda_fp16.h>

// Warp-level reduction sum using shuffle operations
__device__ __forceinline__ float warpReduceSum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Block-level reduction using shared memory
template<int BLOCK_SIZE>
__device__ __forceinline__ float blockReduceSum(float val) {
    __shared__ float shared[32]; // One element per warp
    
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;
    
    // Each warp performs its own reduction
    val = warpReduceSum(val);
    
    // Write reduced value to shared memory
    if (lane == 0) {
        shared[wid] = val;
    }
    __syncthreads();
    
    // Read from shared memory only if that warp existed
    val = (threadIdx.x < (BLOCK_SIZE / 32)) ? shared[lane] : 0.0f;
    
    // Final reduce within first warp
    if (wid == 0) {
        val = warpReduceSum(val);
    }
    
    return val;
}

// Optimized kernel for mean reduction along dimension 0
// Each block handles one output element
template<int BLOCK_SIZE>
__global__ void mean_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int rows,
    int inner
) {
    int col = blockIdx.x;
    if (col >= inner) return;
    
    float sum = 0.0f;
    
    // Grid-stride loop to accumulate all rows for this column
    for (int row = threadIdx.x; row < rows; row += BLOCK_SIZE) {
        sum += input[row * inner + col];
    }
    
    // Block-level reduction
    sum = blockReduceSum<BLOCK_SIZE>(sum);
    
    // Write result
    if (threadIdx.x == 0) {
        output[col] = sum / static_cast<float>(rows);
    }
}

// Optimized kernel for small row counts (uses vectorized loads when possible)
__global__ void mean_kernel_small(
    const float* __restrict__ input,
    float* __restrict__ output,
    int rows,
    int inner
) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (col < inner) {
        float sum = 0.0f;
        
        #pragma unroll 8
        for (int row = 0; row < rows; row++) {
            sum += input[row * inner + col];
        }
        
        output[col] = sum / static_cast<float>(rows);
    }
}

// Host function to launch the appropriate kernel
extern "C" void cuda_kernel(
    const float* input,
    float* output,
    int rows,
    int inner
) {
    if (rows <= 0 || inner <= 0) return;
    
    // Choose kernel based on problem size
    if (rows <= 16) {
        // For small row counts, use simple kernel with vectorized access
        int threads = 256;
        int blocks = (inner + threads - 1) / threads;
        mean_kernel_small<<<blocks, threads>>>(input, output, rows, inner);
    } else if (rows <= 64) {
        // Medium row count - use 128 threads per block
        int blocks = inner;
        mean_kernel<128><<<blocks, 128>>>(input, output, rows, inner);
    } else {
        // Large row count - use 256 threads per block
        int blocks = inner;
        mean_kernel<256><<<blocks, 256>>>(input, output, rows, inner);
    }
}
