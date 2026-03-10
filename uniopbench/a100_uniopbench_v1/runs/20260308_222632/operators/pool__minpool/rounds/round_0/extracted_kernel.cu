#include <cuda_runtime.h>
#include <float.h>

// MinPool kernel for NHWC layout
// Each thread computes one output element
__global__ void minpool_kernel(
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
    // Global thread index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Total number of output elements
    int total_output = batch_size * output_H * output_W * channels;
    
    // Grid-stride loop for handling all output elements
    for (int i = idx; i < total_output; i += blockDim.x * gridDim.x) {
        // Decode output position from flat index (NHWC layout)
        int c = i % channels;
        int w_out = (i / channels) % output_W;
        int h_out = (i / (channels * output_W)) % output_H;
        int n = i / (channels * output_W * output_H);
        
        // Starting position in input
        int h_start = h_out * stride;
        int w_start = w_out * stride;
        
        // Initialize with maximum float value for min operation
        float min_val = FLT_MAX;
        
        // Scan the pooling window
        for (int kh = 0; kh < kernel_size; kh++) {
            for (int kw = 0; kw < kernel_size; kw++) {
                int h_in = h_start + kh;
                int w_in = w_start + kw;
                
                // Bounds check
                if (h_in < input_H && w_in < input_W) {
                    // NHWC layout: [batch][height][width][channel]
                    int input_idx = ((n * input_H + h_in) * input_W + w_in) * channels + c;
                    float val = input[input_idx];
                    min_val = fminf(min_val, val);
                }
            }
        }
        
        // Write output
        output[i] = min_val;
    }
}

extern "C" void cuda_kernel(
    const float* input,
    float* output,
    int kernel_size,
    int stride,
    int batch_size,
    int channels,
    int input_H
) {
    // Calculate output dimensions
    int input_W = input_H;  // Assuming square input
    int output_H = (input_H - kernel_size) / stride + 1;
    int output_W = output_H;  // Square output
    
    // Calculate total output elements
    int total_output = batch_size * output_H * output_W * channels;
    
    // Configure kernel launch parameters
    // Use 256 threads per block (good for A100)
    int threads_per_block = 256;
    // Calculate number of blocks needed
    int num_blocks = (total_output + threads_per_block - 1) / threads_per_block;
    // Cap the number of blocks for efficiency
    num_blocks = min(num_blocks, 2048);
    
    // Launch kernel
    minpool_kernel<<<num_blocks, threads_per_block>>>(
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
}
