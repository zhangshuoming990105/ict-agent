#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>
#include <algorithm>

template<typename T>
__device__ __forceinline__ T load_value(const T* ptr) {
    return *ptr;
}

template<typename T>
__device__ __forceinline__ void store_value(T* ptr, T value) {
    *ptr = value;
}

// Specialization for half precision
__device__ __forceinline__ float load_value(const __half* ptr) {
    return __half2float(*ptr);
}

__device__ __forceinline__ void store_value(__half* ptr, float value) {
    *ptr = __float2half(value);
}

template<typename T>
__global__ void merge_attn_states_kernel(
    T* output,                      // [num_tokens, num_heads, head_size]
    float* output_lse,              // [num_heads, num_tokens]
    const T* prefix_output,         // [num_tokens, num_heads, head_size]
    const float* prefix_lse,        // [num_heads, num_tokens]
    const T* suffix_output,         // [num_tokens, num_heads, head_size]
    const float* suffix_lse,        // [num_heads, num_tokens]
    int num_tokens,
    int num_heads,
    int head_size
) {
    // Grid-stride loop over all elements
    // Each thread processes one output element
    int total_elements = num_tokens * num_heads * head_size;
    
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; 
         idx < total_elements; 
         idx += blockDim.x * gridDim.x) {
        
        // Decode indices: output is [num_tokens, num_heads, head_size]
        int head_dim_idx = idx % head_size;
        int head_idx = (idx / head_size) % num_heads;
        int token_idx = idx / (head_size * num_heads);
        
        // LSE tensors are [num_heads, num_tokens]
        int lse_idx = head_idx * num_tokens + token_idx;
        
        float p_lse = prefix_lse[lse_idx];
        float s_lse = suffix_lse[lse_idx];
        
        // Handle infinity: convert +inf to -inf
        if (isinf(p_lse) && p_lse > 0) p_lse = -INFINITY;
        if (isinf(s_lse) && s_lse > 0) s_lse = -INFINITY;
        
        // Compute max_lse and scales
        float max_lse = fmaxf(p_lse, s_lse);
        float p_se = expf(p_lse - max_lse);
        float s_se = expf(s_lse - max_lse);
        float out_se = p_se + s_se;
        
        float p_scale = p_se / out_se;
        float s_scale = s_se / out_se;
        
        // Load prefix and suffix values
        float p_val = load_value(&prefix_output[idx]);
        float s_val = load_value(&suffix_output[idx]);
        
        // Compute merged output
        float result = p_val * p_scale + s_val * s_scale;
        
        // Store output
        store_value(&output[idx], result);
        
        // Only the first thread for each (head, token) pair writes output_lse
        // We use head_dim_idx == 0 to ensure only one thread does this
        if (head_dim_idx == 0 && output_lse != nullptr) {
            float new_lse = logf(out_se) + max_lse;
            output_lse[lse_idx] = new_lse;
        }
    }
}

extern "C" {

void cuda_kernel(
    void* output_ptr,
    void* output_lse_ptr,
    const void* prefix_output_ptr,
    const void* prefix_lse_ptr,
    const void* suffix_output_ptr,
    const void* suffix_lse_ptr,
    int num_tokens,
    int num_heads,
    int head_size,
    int dtype_code  // 0 = float32, 1 = float16
) {
    int total_elements = num_tokens * num_heads * head_size;
    
    // Configure kernel launch parameters
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;
    blocks = min(blocks, 1024);  // Cap at 1024 blocks
    
    if (dtype_code == 0) {
        // float32
        merge_attn_states_kernel<float><<<blocks, threads>>>(
            static_cast<float*>(output_ptr),
            static_cast<float*>(output_lse_ptr),
            static_cast<const float*>(prefix_output_ptr),
            static_cast<const float*>(prefix_lse_ptr),
            static_cast<const float*>(suffix_output_ptr),
            static_cast<const float*>(suffix_lse_ptr),
            num_tokens,
            num_heads,
            head_size
        );
    } else {
        // float16
        merge_attn_states_kernel<__half><<<blocks, threads>>>(
            static_cast<__half*>(output_ptr),
            static_cast<float*>(output_lse_ptr),
            static_cast<const __half*>(prefix_output_ptr),
            static_cast<const float*>(prefix_lse_ptr),
            static_cast<const __half*>(suffix_output_ptr),
            static_cast<const float*>(suffix_lse_ptr),
            num_tokens,
            num_heads,
            head_size
        );
    }
}

}  // extern "C"
