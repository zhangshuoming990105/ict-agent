#include <cuda_runtime.h>
#include <cstdint>

// First copy input to output
__global__ void copy_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int total_elements
) {
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; 
         idx < total_elements; 
         idx += blockDim.x * gridDim.x) {
        output[idx] = input[idx];
    }
}

// Scatter kernel along dimension 3 (W dimension)
// For each element at (n, c, h, w), write it to output[n, c, h, indices[n,c,h,w]]
__global__ void scatter_kernel(
    const float* __restrict__ input,
    const int* __restrict__ indices,
    float* __restrict__ output,
    int N, int C, int H, int W
) {
    // Total elements to process
    int total_elements = N * C * H * W;
    
    // Grid-stride loop
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; 
         idx < total_elements; 
         idx += blockDim.x * gridDim.x) {
        
        // Decompose linear index into (n, c, h, w)
        int w = idx % W;
        int h = (idx / W) % H;
        int c = (idx / (W * H)) % C;
        int n = idx / (W * H * C);
        
        // Read the index to scatter to
        int scatter_w = indices[idx];
        
        // Compute output index: (n, c, h, scatter_w)
        int out_idx = n * (C * H * W) + c * (H * W) + h * W + scatter_w;
        
        // Write the value from input to the scattered position in output
        output[out_idx] = input[idx];
    }
}

extern "C" {
    void cuda_kernel(
        const float* input,
        const int* indices,
        float* output,
        int N, int C, int H, int W
    ) {
        int total_elements = N * C * H * W;
        
        // Launch configuration
        int threads = 256;
        int blocks = (total_elements + threads - 1) / threads;
        blocks = min(blocks, 1024);
        
        // First copy input to output
        copy_kernel<<<blocks, threads>>>(input, output, total_elements);
        
        // Then perform scatter
        scatter_kernel<<<blocks, threads>>>(input, indices, output, N, C, H, W);
    }
}
