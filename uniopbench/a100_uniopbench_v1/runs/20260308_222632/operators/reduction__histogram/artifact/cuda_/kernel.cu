#include <cuda_runtime.h>
#include <stdexcept>

// Histogram kernel using atomic operations
// op_type: 0 = i32 (scalar), 1 = i32x4 (vectorized)
__global__ void histogram_kernel_i32(const int* __restrict__ input, 
                                      int* __restrict__ output,
                                      int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    
    for (int i = idx; i < n; i += stride) {
        int val = input[i];
        atomicAdd(&output[val], 1);
    }
}

__global__ void histogram_kernel_i32x4(const int* __restrict__ input,
                                        int* __restrict__ output,
                                        int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    
    // Process 4 elements per thread when possible
    int vec_n = n / 4 * 4;
    
    for (int i = idx * 4; i < vec_n; i += stride * 4) {
        int4 vals = reinterpret_cast<const int4*>(input)[i / 4];
        atomicAdd(&output[vals.x], 1);
        atomicAdd(&output[vals.y], 1);
        atomicAdd(&output[vals.z], 1);
        atomicAdd(&output[vals.w], 1);
    }
    
    // Handle remaining elements
    for (int i = vec_n + idx; i < n; i += stride) {
        int val = input[i];
        atomicAdd(&output[val], 1);
    }
}

extern "C" {

void cuda_kernel(const int* input, int* output, int n, int op_type) {
    // Validate inputs
    if (input == nullptr || output == nullptr) {
        throw std::runtime_error("Null pointer passed to histogram kernel");
    }
    
    if (n <= 0) {
        return;  // Nothing to do
    }
    
    // Launch configuration
    const int threads = 256;
    const int blocks = min((n + threads - 1) / threads, 1024);
    
    if (op_type == 0) {
        // Scalar i32 version
        histogram_kernel_i32<<<blocks, threads>>>(input, output, n);
    } else {
        // Vectorized i32x4 version
        histogram_kernel_i32x4<<<blocks, threads>>>(input, output, n);
    }
    
    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA kernel launch failed: ") + 
                                cudaGetErrorString(err));
    }
}

}
