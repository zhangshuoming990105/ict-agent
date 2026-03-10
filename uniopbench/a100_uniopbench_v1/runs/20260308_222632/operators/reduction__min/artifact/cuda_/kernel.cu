#include <cuda_runtime.h>
#include <cfloat>

// Warp-level reduction for minimum
__device__ __forceinline__ float warpReduceMin(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val = fminf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

// Block-level reduction for minimum
__device__ __forceinline__ float blockReduceMin(float val) {
    static __shared__ float shared[32]; // One element per warp
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;

    // Each warp performs partial reduction
    val = warpReduceMin(val);

    // Write reduced value to shared memory
    if (lane == 0) {
        shared[wid] = val;
    }
    __syncthreads();

    // Read from shared memory only if that warp existed
    int numWarps = (blockDim.x + 31) / 32;
    val = (threadIdx.x < numWarps) ? shared[lane] : FLT_MAX;

    // Final reduce within first warp
    if (wid == 0) {
        val = warpReduceMin(val);
    }

    return val;
}

// Kernel for computing minimum reduction along dimension 0
// Input shape: (rows, inner)
// Output shape: (inner,)
__global__ void min_reduction_kernel(const float* __restrict__ input,
                                      float* __restrict__ output,
                                      int rows,
                                      int inner) {
    // Each block processes one or more columns
    for (int col = blockIdx.x; col < inner; col += gridDim.x) {
        // Each thread computes partial min for this column
        float thread_min = FLT_MAX;
        
        // Grid-stride loop over rows
        for (int row = threadIdx.x; row < rows; row += blockDim.x) {
            int idx = row * inner + col;
            thread_min = fminf(thread_min, input[idx]);
        }
        
        // Reduce within block
        float block_min = blockReduceMin(thread_min);
        
        // Thread 0 writes result
        if (threadIdx.x == 0) {
            output[col] = block_min;
        }
    }
}

// Host interface function
extern "C" void cuda_kernel(const float* input, float* output, int rows, int inner) {
    // Choose grid and block dimensions
    int threadsPerBlock = 256;
    int blocksPerGrid = min(inner, 1024); // Limit blocks for better occupancy
    
    min_reduction_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        input, output, rows, inner
    );
}
