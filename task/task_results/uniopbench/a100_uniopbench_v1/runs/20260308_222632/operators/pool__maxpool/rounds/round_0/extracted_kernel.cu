#include <cuda_runtime.h>
#include <float.h>

// Optimized MaxPool2D kernel for A100
// Input/Output format: NCHW contiguous layout
// Each thread processes one output element

__global__ void maxpool2d_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int batch_size,
    int channels,
    int input_H,
    int input_W,
    int output_H,
    int output_W,
    int kernel_size,
    int stride
) {
    // Global thread index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Total number of output elements
    int total_elements = batch_size * channels * output_H * output_W;
    
    // Grid-stride loop for better SM utilization
    for (int i = idx; i < total_elements; i += blockDim.x * gridDim.x) {
        // Decode output position (NCHW layout)
        int w_out = i % output_W;
        int h_out = (i / output_W) % output_H;
        int c = (i / (output_W * output_H)) % channels;
        int n = i / (channels * output_H * output_W);
        
        // Calculate input starting position
        int h_start = h_out * stride;
        int w_start = w_out * stride;
        
        // Initialize max value
        float max_val = -FLT_MAX;
        
        // Perform max pooling over the kernel window
        #pragma unroll 5
        for (int kh = 0; kh < kernel_size; kh++) {
            int h_in = h_start + kh;
            if (h_in < input_H) {
                #pragma unroll 5
                for (int kw = 0; kw < kernel_size; kw++) {
                    int w_in = w_start + kw;
                    if (w_in < input_W) {
                        // Calculate input index in NCHW layout
                        int input_idx = ((n * channels + c) * input_H + h_in) * input_W + w_in;
                        float val = input[input_idx];
                        max_val = fmaxf(max_val, val);
                    }
                }
            }
        }
        
        // Write output
        output[i] = max_val;
    }
}

extern "C" {

void cuda_kernel(
    const float* input,
    float* output,
    int kernel_size,
    int stride,
    int batch_size,
    int channels,
    int input_H
) {
    // Calculate output dimensions
    int input_W = input_H;  // Assuming square input based on test.py
    int output_H = (input_H - kernel_size) / stride + 1;
    int output_W = output_H;  // Square output
    
    // Calculate total output elements
    int total_elements = batch_size * output_H * output_W * channels;
    
    // Launch configuration
    const int threads_per_block = 256;
    int num_blocks = (total_elements + threads_per_block - 1) / threads_per_block;
    
    // Limit number of blocks for better occupancy
    num_blocks = min(num_blocks, 2048);
    
    maxpool2d_kernel<<<num_blocks, threads_per_block>>>(
        input,
        output,
        batch_size,
        channels,
        input_H,
        input_W,
        output_H,
        output_W,
        kernel_size,
        stride
    );
}

}
