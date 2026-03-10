#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdexcept>

// CUDA kernel for SumPool operation in NHWC format
// This implements sum pooling which is equivalent to avg_pool * (kernel_size^2)
// Input/Output format: NHWC (batch, height, width, channels)

__global__ void sumpool_kernel(
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
    // Grid-stride loop pattern for better occupancy
    // Each thread computes one output element
    const int total_output = batch_size * output_H * output_W * channels;
    
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < total_output;
         idx += blockDim.x * gridDim.x) {
        
        // Decode NHWC indices
        const int c = idx % channels;
        const int w_out = (idx / channels) % output_W;
        const int h_out = (idx / channels / output_W) % output_H;
        const int n = idx / (channels * output_W * output_H);
        
        // Calculate input window position
        const int h_in_start = h_out * stride;
        const int w_in_start = w_out * stride;
        
        // Accumulate sum over the pooling window
        float sum = 0.0f;
        
        for (int kh = 0; kh < kernel_size; ++kh) {
            const int h_in = h_in_start + kh;
            
            for (int kw = 0; kw < kernel_size; ++kw) {
                const int w_in = w_in_start + kw;
                
                // NHWC layout: [n, h, w, c]
                const int input_idx = ((n * input_H + h_in) * input_W + w_in) * channels + c;
                sum += input[input_idx];
            }
        }
        
        // Write result
        output[idx] = sum;
    }
}

// Host function to launch the kernel
extern "C" void cuda_kernel(
    const float* input,
    float* output,
    int batch_size,
    int channels,
    int input_H,
    int kernel_size,
    int stride
) {
    // Calculate output dimensions (assuming square input and kernel)
    int input_W = input_H;  // Square input
    int output_H = (input_H - kernel_size) / stride + 1;
    int output_W = (input_W - kernel_size) / stride + 1;
    
    // Calculate total output elements
    int total_output = batch_size * output_H * output_W * channels;
    
    // Launch configuration optimized for A100
    // Use 256 threads per block for good occupancy
    const int threads_per_block = 256;
    const int blocks = (total_output + threads_per_block - 1) / threads_per_block;
    
    // Limit grid size to avoid excessive unused blocks
    const int max_blocks = 2048;
    const int grid_size = min(blocks, max_blocks);
    
    // Launch kernel
    sumpool_kernel<<<grid_size, threads_per_block>>>(
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
    
    // Check for launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("CUDA kernel launch failed: " + 
                                 std::string(cudaGetErrorString(err)));
    }
}
