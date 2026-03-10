#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// Depthwise Convolution CUDA Kernel (HWC format)
// Optimized for A100 (sm_80) with coalesced memory access
// Each thread computes one output pixel across all its channels

template<int KERNEL_SIZE>
__global__ void depthwise_conv_kernel_template(
    const float* __restrict__ input,
    const float* __restrict__ kernel,
    float* __restrict__ output,
    int input_height,
    int channels)
{
    int output_size = input_height - KERNEL_SIZE + 1;
    int total_elements = output_size * output_size * channels;
    
    // Thread index maps to output position
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Grid-stride loop for efficient processing
    for (int i = idx; i < total_elements; i += blockDim.x * gridDim.x) {
        // Decode output position: HWC layout
        int c = i % channels;
        int w = (i / channels) % output_size;
        int h = i / (channels * output_size);
        
        float sum = 0.0f;
        
        // Unrolled convolution loop for small kernels
        #pragma unroll
        for (int kh = 0; kh < KERNEL_SIZE; kh++) {
            #pragma unroll
            for (int kw = 0; kw < KERNEL_SIZE; kw++) {
                int input_h = h + kh;
                int input_w = w + kw;
                
                // HWC layout indexing - channel is innermost dimension for coalescing
                int input_idx = (input_h * input_height + input_w) * channels + c;
                int kernel_idx = (kh * KERNEL_SIZE + kw) * channels + c;
                
                sum = __fmaf_rn(input[input_idx], kernel[kernel_idx], sum);
            }
        }
        
        output[i] = sum;
    }
}

// Generic kernel for arbitrary kernel sizes
__global__ void depthwise_conv_kernel_generic(
    const float* __restrict__ input,
    const float* __restrict__ kernel,
    float* __restrict__ output,
    int input_height,
    int kernel_size,
    int channels)
{
    int output_size = input_height - kernel_size + 1;
    int total_elements = output_size * output_size * channels;
    
    // Thread index maps to output position
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Grid-stride loop for efficient processing
    for (int i = idx; i < total_elements; i += blockDim.x * gridDim.x) {
        // Decode output position: HWC layout
        int c = i % channels;
        int w = (i / channels) % output_size;
        int h = i / (channels * output_size);
        
        float sum = 0.0f;
        
        // Convolution loop
        for (int kh = 0; kh < kernel_size; kh++) {
            for (int kw = 0; kw < kernel_size; kw++) {
                int input_h = h + kh;
                int input_w = w + kw;
                
                // HWC layout indexing
                int input_idx = (input_h * input_height + input_w) * channels + c;
                int kernel_idx = (kh * kernel_size + kw) * channels + c;
                
                sum = __fmaf_rn(input[input_idx], kernel[kernel_idx], sum);
            }
        }
        
        output[i] = sum;
    }
}

// Host function that launches the appropriate kernel
extern "C" void cuda_kernel(
    const float* input,
    const float* kernel,
    float* output,
    int input_height,
    int kernel_size,
    int channels)
{
    int output_size = input_height - kernel_size + 1;
    int total_elements = output_size * output_size * channels;
    
    // Configure launch parameters
    // Use 256 threads per block (good occupancy on A100)
    int threads_per_block = 256;
    int num_blocks = (total_elements + threads_per_block - 1) / threads_per_block;
    
    // Limit number of blocks for very large problems to avoid overwhelming the GPU
    num_blocks = min(num_blocks, 2048);
    
    // Dispatch to specialized kernel for common kernel sizes
    if (kernel_size == 3) {
        depthwise_conv_kernel_template<3><<<num_blocks, threads_per_block>>>(
            input, kernel, output, input_height, channels);
    } else if (kernel_size == 5) {
        depthwise_conv_kernel_template<5><<<num_blocks, threads_per_block>>>(
            input, kernel, output, input_height, channels);
    } else if (kernel_size == 7) {
        depthwise_conv_kernel_template<7><<<num_blocks, threads_per_block>>>(
            input, kernel, output, input_height, channels);
    } else {
        // Fallback to generic kernel
        depthwise_conv_kernel_generic<<<num_blocks, threads_per_block>>>(
            input, kernel, output, input_height, kernel_size, channels);
    }
}
