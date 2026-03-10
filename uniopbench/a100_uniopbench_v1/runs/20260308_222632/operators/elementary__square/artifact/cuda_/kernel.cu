#include <cuda_runtime.h>

// CUDA kernel for element-wise square operation
__global__ void square_kernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    // Grid-stride loop to handle any size
    for (int i = idx; i < size; i += stride) {
        float val = input[i];
        output[i] = val * val;
    }
}

// C interface for the kernel launcher
extern "C" {
    void cuda_kernel(const float* input, float* output, int size) {
        // Choose block size
        const int blockSize = 256;
        
        // Calculate grid size to cover all elements
        // Use at least enough blocks to cover all elements, but cap at a reasonable maximum
        int numBlocks = (size + blockSize - 1) / blockSize;
        numBlocks = min(numBlocks, 65535); // Cap at max grid dimension
        
        // Launch kernel
        square_kernel<<<numBlocks, blockSize>>>(input, output, size);
        
        // Note: synchronization is handled by the backend
    }
}
