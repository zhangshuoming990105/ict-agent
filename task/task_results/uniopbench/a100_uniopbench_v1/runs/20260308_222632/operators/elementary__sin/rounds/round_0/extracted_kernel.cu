#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>
#include <cstdint>

// Vectorized sin kernel for float32
// Uses float4 vectorization for coalesced memory access
__global__ void sin_kernel_float_vectorized(const float* __restrict__ x,
                                            float* __restrict__ output,
                                            int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    // Vectorized loads when possible (process 4 elements at a time)
    int vec_size = size / 4;
    const float4* x_vec = reinterpret_cast<const float4*>(x);
    float4* output_vec = reinterpret_cast<float4*>(output);
    
    for (int i = idx; i < vec_size; i += stride) {
        float4 val = x_vec[i];
        float4 result;
        result.x = sinf(val.x);
        result.y = sinf(val.y);
        result.z = sinf(val.z);
        result.w = sinf(val.w);
        output_vec[i] = result;
    }
    
    // Handle remaining elements
    int base = vec_size * 4;
    for (int i = base + idx; i < size; i += stride) {
        output[i] = sinf(x[i]);
    }
}

// Standard sin kernel for float32
__global__ void sin_kernel_float(const float* __restrict__ x,
                                 float* __restrict__ output,
                                 int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    for (int i = idx; i < size; i += stride) {
        output[i] = sinf(x[i]);
    }
}

// Vectorized sin kernel for float16
// Uses half2 vectorization for better memory bandwidth
__global__ void sin_kernel_half_vectorized(const half* __restrict__ x,
                                           half* __restrict__ output,
                                           int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    // Vectorized loads (process 2 half elements at a time)
    int vec_size = size / 2;
    const half2* x_vec = reinterpret_cast<const half2*>(x);
    half2* output_vec = reinterpret_cast<half2*>(output);
    
    for (int i = idx; i < vec_size; i += stride) {
        half2 val = x_vec[i];
        half2 result;
        result.x = hsin(val.x);
        result.y = hsin(val.y);
        output_vec[i] = result;
    }
    
    // Handle remaining element if size is odd
    if (size % 2 == 1) {
        int last_idx = size - 1;
        if (idx == 0) {
            output[last_idx] = hsin(x[last_idx]);
        }
    }
}

// Standard sin kernel for float16
__global__ void sin_kernel_half(const half* __restrict__ x,
                                half* __restrict__ output,
                                int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    for (int i = idx; i < size; i += stride) {
        output[i] = hsin(x[i]);
    }
}

// Host function that dispatches based on dtype_id
// Expected function name by optest framework
extern "C" {

void cuda_kernel(void* x, void* output, int size, int dtype_id) {
    // Determine optimal grid/block dimensions
    const int block_size = 256;
    const int num_blocks = min((size + block_size - 1) / block_size, 1024);
    
    if (dtype_id == 0) {
        // float32
        // Check if pointers are aligned for vectorized access
        uintptr_t x_addr = reinterpret_cast<uintptr_t>(x);
        uintptr_t output_addr = reinterpret_cast<uintptr_t>(output);
        bool aligned = (x_addr % 16 == 0) && (output_addr % 16 == 0);
        
        if (aligned && size >= 4) {
            sin_kernel_float_vectorized<<<num_blocks, block_size>>>(
                static_cast<const float*>(x),
                static_cast<float*>(output),
                size
            );
        } else {
            sin_kernel_float<<<num_blocks, block_size>>>(
                static_cast<const float*>(x),
                static_cast<float*>(output),
                size
            );
        }
    } else if (dtype_id == 1) {
        // float16
        // Check if pointers are aligned for vectorized access
        uintptr_t x_addr = reinterpret_cast<uintptr_t>(x);
        uintptr_t output_addr = reinterpret_cast<uintptr_t>(output);
        bool aligned = (x_addr % 4 == 0) && (output_addr % 4 == 0);
        
        if (aligned && size >= 2) {
            sin_kernel_half_vectorized<<<num_blocks, block_size>>>(
                static_cast<const half*>(x),
                static_cast<half*>(output),
                size
            );
        } else {
            sin_kernel_half<<<num_blocks, block_size>>>(
                static_cast<const half*>(x),
                static_cast<half*>(output),
                size
            );
        }
    }
    // If dtype_id is neither 0 nor 1, do nothing (could add error handling)
}

} // extern "C"
