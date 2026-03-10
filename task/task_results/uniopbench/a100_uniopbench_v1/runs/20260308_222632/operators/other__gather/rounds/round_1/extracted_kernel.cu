#include <cuda_runtime.h>
#include <stdint.h>

// CUDA kernel for gather operation along last dimension
// Input: params[dim0, dim1, dim2]
// Indices: indices[num_indices]
// Output: output[dim0, dim1, num_indices]
__global__ void gather_kernel_impl(
    const float* __restrict__ params,
    const int64_t* __restrict__ indices,
    float* __restrict__ output,
    int dim0,
    int dim1,
    int dim2,
    int num_indices
) {
    // Total output elements: dim0 * dim1 * num_indices
    int total_elements = dim0 * dim1 * num_indices;
    
    // Use grid-stride loop for handling arbitrary sizes
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < total_elements;
         idx += blockDim.x * gridDim.x) {
        
        // Decompose linear index into (i, j, k) where:
        // i = batch dimension (0..dim0-1)
        // j = middle dimension (0..dim1-1)
        // k = output index dimension (0..num_indices-1)
        int k = idx % num_indices;
        int temp = idx / num_indices;
        int j = temp % dim1;
        int i = temp / dim1;
        
        // Read the index value (which dimension in dim2 to gather from)
        int64_t gather_idx = indices[k];
        
        // Calculate input position: params[i, j, gather_idx]
        // Input is laid out as [dim0, dim1, dim2]
        int input_idx = (i * dim1 + j) * dim2 + static_cast<int>(gather_idx);
        
        // Write to output[i, j, k]
        output[idx] = params[input_idx];
    }
}

// Host interface function - named cuda_kernel as expected by optest framework
extern "C" void cuda_kernel(
    const float* params,
    const int64_t* indices,
    float* output,
    int dim0,
    int dim1,
    int dim2,
    int num_indices
) {
    int total_elements = dim0 * dim1 * num_indices;
    
    // Configure launch parameters
    int threads_per_block = 256;
    int num_blocks = (total_elements + threads_per_block - 1) / threads_per_block;
    
    // Limit grid size for efficiency
    num_blocks = min(num_blocks, 2048);
    
    gather_kernel_impl<<<num_blocks, threads_per_block>>>(
        params,
        indices,
        output,
        dim0,
        dim1,
        dim2,
        num_indices
    );
}
