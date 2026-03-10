#include <cuda_runtime.h>
#include <cstdint>

// Optimized concat kernel for A100 (sm_80)
// Concatenates two tensors along channel dimension
// Input1: (N, C, H, W)
// Input2: (N, C, H, W)
// Output: (N, 2*C, H, W)

// Vectorized kernel using float4 for coalesced memory access
__global__ void concat_kernel_vec4(
    const float* __restrict__ input1,
    const float* __restrict__ input2,
    float* __restrict__ output,
    int N, int C, int H, int W)
{
    const int spatial_size = H * W;
    const int channel_spatial_size = C * spatial_size;
    const int output_channel_spatial_size = 2 * C * spatial_size;
    
    // Each thread processes 4 floats
    const int vec_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_vec_elements = (N * C * spatial_size) / 4;
    
    if (vec_idx < total_vec_elements) {
        // Linear index in float elements
        const int linear_idx = vec_idx * 4;
        
        // Calculate batch and position within batch
        const int batch_idx = linear_idx / channel_spatial_size;
        const int within_batch = linear_idx % channel_spatial_size;
        const int channel_idx = within_batch / spatial_size;
        const int spatial_idx = within_batch % spatial_size;
        
        // Check if we can do aligned float4 access
        if ((spatial_idx % 4 == 0) && (spatial_idx + 4 <= spatial_size)) {
            // Vectorized load from input1
            const int src_offset = batch_idx * channel_spatial_size + 
                                   channel_idx * spatial_size + spatial_idx;
            float4 data1 = *reinterpret_cast<const float4*>(input1 + src_offset);
            
            // Vectorized load from input2
            float4 data2 = *reinterpret_cast<const float4*>(input2 + src_offset);
            
            // Vectorized store to output - first C channels
            const int dst_offset1 = batch_idx * output_channel_spatial_size + 
                                    channel_idx * spatial_size + spatial_idx;
            *reinterpret_cast<float4*>(output + dst_offset1) = data1;
            
            // Vectorized store to output - second C channels
            const int dst_offset2 = batch_idx * output_channel_spatial_size + 
                                    (C + channel_idx) * spatial_size + spatial_idx;
            *reinterpret_cast<float4*>(output + dst_offset2) = data2;
        } else {
            // Scalar fallback for misaligned access
            for (int i = 0; i < 4 && (linear_idx + i) < N * channel_spatial_size; i++) {
                const int idx = linear_idx + i;
                const int b = idx / channel_spatial_size;
                const int wb = idx % channel_spatial_size;
                const int c = wb / spatial_size;
                const int s = wb % spatial_size;
                
                const int dst1 = b * output_channel_spatial_size + c * spatial_size + s;
                const int dst2 = b * output_channel_spatial_size + (C + c) * spatial_size + s;
                
                output[dst1] = input1[idx];
                output[dst2] = input2[idx];
            }
        }
    }
}

// Simple scalar kernel for fallback
__global__ void concat_kernel_scalar(
    const float* __restrict__ input1,
    const float* __restrict__ input2,
    float* __restrict__ output,
    int N, int C, int H, int W)
{
    const int spatial_size = H * W;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_elements = N * C * spatial_size;
    
    if (idx < total_elements) {
        const int channel_spatial_size = C * spatial_size;
        const int batch_idx = idx / channel_spatial_size;
        const int within_batch = idx % channel_spatial_size;
        
        const int channel_idx = within_batch / spatial_size;
        const int spatial_idx = within_batch % spatial_size;
        
        const int output_channel_spatial_size = 2 * C * spatial_size;
        
        // Copy from input1 to first C channels of output
        const int dst_offset1 = batch_idx * output_channel_spatial_size + 
                                channel_idx * spatial_size + spatial_idx;
        output[dst_offset1] = input1[idx];
        
        // Copy from input2 to second C channels of output
        const int dst_offset2 = batch_idx * output_channel_spatial_size + 
                                (C + channel_idx) * spatial_size + spatial_idx;
        output[dst_offset2] = input2[idx];
    }
}

extern "C" void cuda_kernel(
    const float* input1,
    const float* input2,
    float* output,
    int N, int C, int H, int W)
{
    const int spatial_size = H * W;
    const int total_elements = N * C * spatial_size;
    
    const int block_size = 256;
    
    // Use vectorized kernel if spatial dimension is divisible by 4
    if (spatial_size % 4 == 0) {
        const int vec_elements = total_elements / 4;
        const int num_blocks = (vec_elements + block_size - 1) / block_size;
        concat_kernel_vec4<<<num_blocks, block_size>>>(
            input1, input2, output, N, C, H, W);
    } else {
        // Fall back to scalar kernel
        const int num_blocks = (total_elements + block_size - 1) / block_size;
        concat_kernel_scalar<<<num_blocks, block_size>>>(
            input1, input2, output, N, C, H, W);
    }
}
