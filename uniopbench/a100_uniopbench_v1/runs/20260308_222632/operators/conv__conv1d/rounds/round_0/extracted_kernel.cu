// CUDA kernel for 1D convolution
// Computes: output[i] = sum(input[i+j] * kernel[j]) for j = 0..kernel_size-1
// where kernel_size = input_size - output_size + 1

#include <cuda_runtime.h>

// Optimized kernel using shared memory for the convolution kernel
__global__ void conv1d_kernel(const float* __restrict__ input,
                              const float* __restrict__ kernel,
                              float* __restrict__ output,
                              int input_size,
                              int output_size,
                              int kernel_size) {
    // Shared memory for kernel weights
    extern __shared__ float s_kernel[];
    
    // Load kernel into shared memory cooperatively
    for (int i = threadIdx.x; i < kernel_size; i += blockDim.x) {
        s_kernel[i] = kernel[i];
    }
    __syncthreads();
    
    // Grid-stride loop for output computation
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    for (int out_idx = idx; out_idx < output_size; out_idx += blockDim.x * gridDim.x) {
        float sum = 0.0f;
        
        // Compute convolution at this output position
        #pragma unroll 4
        for (int k = 0; k < kernel_size; k++) {
            sum += input[out_idx + k] * s_kernel[k];
        }
        
        output[out_idx] = sum;
    }
}

// Host interface function - must be named "cuda_kernel" for the framework
extern "C" void cuda_kernel(float* input, float* kernel, float* output,
                            int input_size, int output_size) {
    // Calculate kernel size
    int kernel_size = input_size - output_size + 1;
    
    // Choose optimal block and grid size
    int block_size = 256;
    int grid_size = (output_size + block_size - 1) / block_size;
    
    // Limit grid size for better occupancy
    grid_size = min(grid_size, 256);
    
    // Shared memory size for kernel weights
    int shared_mem_size = kernel_size * sizeof(float);
    
    // Launch kernel
    conv1d_kernel<<<grid_size, block_size, shared_mem_size>>>(
        input, kernel, output, input_size, output_size, kernel_size
    );
}
