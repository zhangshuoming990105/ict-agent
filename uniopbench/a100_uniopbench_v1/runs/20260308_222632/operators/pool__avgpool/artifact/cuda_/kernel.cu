#include <cuda_runtime.h>
#include <cstdio>

// CUDA kernel for 2D average pooling with NCHW layout (contiguous)
// This is the standard PyTorch memory layout for 4D tensors
__global__ void avgpool_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int batch_size,
    int channels,
    int input_H,
    int input_W,
    int kernel_size,
    int stride,
    int output_H,
    int output_W
) {
    // Each thread processes one output element
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * channels * output_H * output_W;
    
    if (idx >= total_elements) return;
    
    // Decode indices for NCHW layout
    // Linear index = n * (C * H * W) + c * (H * W) + h * W + w
    int w_out = idx % output_W;
    int temp = idx / output_W;
    int h_out = temp % output_H;
    temp = temp / output_H;
    int c = temp % channels;
    int n = temp / channels;
    
    // Starting position in input
    int h_in_start = h_out * stride;
    int w_in_start = w_out * stride;
    
    // Accumulate sum over the pooling window
    float sum = 0.0f;
    
    // Input base index for this batch and channel
    int input_base = (n * channels + c) * (input_H * input_W);
    
    #pragma unroll
    for (int kh = 0; kh < 5; ++kh) {
        if (kh >= kernel_size) break;
        int h_in = h_in_start + kh;
        if (h_in >= input_H) break;
        
        #pragma unroll
        for (int kw = 0; kw < 5; ++kw) {
            if (kw >= kernel_size) break;
            int w_in = w_in_start + kw;
            if (w_in >= input_W) break;
            
            // NCHW layout: [n, c, h, w]
            int input_idx = input_base + h_in * input_W + w_in;
            sum += input[input_idx];
        }
    }
    
    // Compute average (kernel_size x kernel_size elements)
    float pool_size = float(kernel_size * kernel_size);
    output[idx] = sum / pool_size;
}

extern "C" {

void cuda_kernel(
    float* input,
    float* output,
    int batch_size,
    int channels,
    int input_H,
    int kernel_size,
    int stride
) {
    // Assume square input (input_W == input_H) and square kernel
    int input_W = input_H;
    int output_H = (input_H - kernel_size) / stride + 1;
    int output_W = (input_W - kernel_size) / stride + 1;
    
    int total_elements = batch_size * channels * output_H * output_W;
    
    // Configure kernel launch
    int threads_per_block = 256;
    int num_blocks = (total_elements + threads_per_block - 1) / threads_per_block;
    
    avgpool_kernel<<<num_blocks, threads_per_block>>>(
        input,
        output,
        batch_size,
        channels,
        input_H,
        input_W,
        kernel_size,
        stride,
        output_H,
        output_W
    );
    
    // Optional: check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA kernel launch error: %s\n", cudaGetErrorString(err));
    }
}

} // extern "C"
