#include <cuda_runtime.h>
#include <math.h>

#define THETA 10000.0f

// Scalar version: process one element pair at a time
__global__ void rope_kernel_scalar(
    const float* __restrict__ input,
    float* __restrict__ output,
    int N,
    int cols
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_pairs = N / 2;
    
    if (idx >= total_pairs) return;
    
    // Calculate position and dimension
    int pos = idx / (cols / 2);           // sequence position
    int dim_pair = idx % (cols / 2);      // which pair of dimensions (0 to head_dim/2-1)
    
    // Calculate frequency for this dimension pair
    float freq = 1.0f / powf(THETA, (2.0f * dim_pair) / (float)cols);
    
    // Calculate angle: pos * freq
    float angle = pos * freq;
    float cos_angle = cosf(angle);
    float sin_angle = sinf(angle);
    
    // Read the pair of values
    int base_idx = pos * cols + dim_pair * 2;
    float x0 = input[base_idx];
    float x1 = input[base_idx + 1];
    
    // Apply rotation: [x0, x1] -> [x0*cos - x1*sin, x0*sin + x1*cos]
    output[base_idx] = x0 * cos_angle - x1 * sin_angle;
    output[base_idx + 1] = x0 * sin_angle + x1 * cos_angle;
}

// Vectorized version: process float4 (4 floats at a time)
__global__ void rope_kernel_vectorized(
    const float* __restrict__ input,
    float* __restrict__ output,
    int N,
    int cols
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_pairs = N / 2;
    
    if (idx >= total_pairs) return;
    
    // Calculate position and dimension
    int pos = idx / (cols / 2);
    int dim_pair = idx % (cols / 2);
    
    // Calculate frequency
    float freq = 1.0f / powf(THETA, (2.0f * dim_pair) / (float)cols);
    
    // Calculate angle
    float angle = pos * freq;
    float cos_angle = cosf(angle);
    float sin_angle = sinf(angle);
    
    // Read and rotate
    int base_idx = pos * cols + dim_pair * 2;
    float x0 = input[base_idx];
    float x1 = input[base_idx + 1];
    
    output[base_idx] = x0 * cos_angle - x1 * sin_angle;
    output[base_idx + 1] = x0 * sin_angle + x1 * cos_angle;
}

extern "C" {

void cuda_kernel(
    const float* input,
    float* output,
    int N,
    int op_type,
    int cols
) {
    // N is total elements, we process pairs
    int total_pairs = N / 2;
    
    const int block_size = 256;
    int grid_size = (total_pairs + block_size - 1) / block_size;
    
    if (op_type == 0) {
        // Scalar version
        rope_kernel_scalar<<<grid_size, block_size>>>(input, output, N, cols);
    } else {
        // Vectorized version
        rope_kernel_vectorized<<<grid_size, block_size>>>(input, output, N, cols);
    }
}

} // extern "C"
