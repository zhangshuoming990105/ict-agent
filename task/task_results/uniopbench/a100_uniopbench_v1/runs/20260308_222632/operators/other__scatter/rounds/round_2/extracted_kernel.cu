#include <cuda_runtime.h>
#include <cstdint>

// Scatter kernel along dimension 3 (W dimension)
// Implements: output.scatter_(dim=3, index=indices, src=input)
// For each position (n, c, h, w), write input[n,c,h,w] to output[n,c,h,indices[n,c,h,w]]
__global__ void scatter_kernel(
    const float* __restrict__ input,
    const int* __restrict__ indices,
    float* __restrict__ output,
    int N, int C, int H, int W
) {
    // Total elements to process
    int total_elements = N * C * H * W;
    
    // Grid-stride loop
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    for (; idx < total_elements; idx += stride) {
        // Decompose linear index into (n, c, h, w)
        int w = idx % W;
        int h = (idx / W) % H;
        int c = (idx / (W * H)) % C;
        int n = idx / (W * H * C);
        
        // Read the index to scatter to (should be in range [0, W))
        int scatter_w = indices[idx];
        
        // Bounds check (safety) - clamp to valid range
        if (scatter_w < 0) scatter_w = 0;
        if (scatter_w >= W) scatter_w = W - 1;
        
        // Compute output index: (n, c, h, scatter_w)
        int out_idx = n * (C * H * W) + c * (H * W) + h * W + scatter_w;
        
        // Write the value from input[n,c,h,w] to output[n,c,h,scatter_w]
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
        
        // Launch configuration - use enough threads
        int threads = 256;
        int blocks = (total_elements + threads - 1) / threads;
        
        // Perform scatter operation directly
        scatter_kernel<<<blocks, threads>>>(input, indices, output, N, C, H, W);
    }
}
