#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

// Sign kernel for float32
__global__ void sign_kernel_float(const float* __restrict__ x, 
                                   float* __restrict__ output, 
                                   int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    // Grid-stride loop for better occupancy and handling arbitrary sizes
    for (int i = idx; i < size; i += stride) {
        float val = x[i];
        // Sign function: -1 if negative, 0 if zero, +1 if positive
        // Using copysignf for branchless computation
        output[i] = (val == 0.0f) ? 0.0f : copysignf(1.0f, val);
    }
}

// Sign kernel for float16
__global__ void sign_kernel_half(const __half* __restrict__ x, 
                                  __half* __restrict__ output, 
                                  int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    const __half zero = __float2half(0.0f);
    const __half one = __float2half(1.0f);
    const __half neg_one = __float2half(-1.0f);
    
    // Grid-stride loop
    for (int i = idx; i < size; i += stride) {
        __half val = x[i];
        float val_f = __half2float(val);
        
        // Sign function for half precision
        __half result;
        if (val_f == 0.0f) {
            result = zero;
        } else if (val_f > 0.0f) {
            result = one;
        } else {
            result = neg_one;
        }
        output[i] = result;
    }
}

// Vectorized sign kernel for float32 using float4
__global__ void sign_kernel_float_vec4(const float* __restrict__ x, 
                                        float* __restrict__ output, 
                                        int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int vec_size = size / 4;
    
    // Vectorized processing
    if (idx < vec_size) {
        float4 val = reinterpret_cast<const float4*>(x)[idx];
        float4 result;
        
        result.x = (val.x == 0.0f) ? 0.0f : copysignf(1.0f, val.x);
        result.y = (val.y == 0.0f) ? 0.0f : copysignf(1.0f, val.y);
        result.z = (val.z == 0.0f) ? 0.0f : copysignf(1.0f, val.z);
        result.w = (val.w == 0.0f) ? 0.0f : copysignf(1.0f, val.w);
        
        reinterpret_cast<float4*>(output)[idx] = result;
    }
    
    // Handle remaining elements
    int remainder_start = vec_size * 4;
    int remainder_idx = remainder_start + idx;
    if (remainder_idx < size) {
        float val = x[remainder_idx];
        output[remainder_idx] = (val == 0.0f) ? 0.0f : copysignf(1.0f, val);
    }
}

// Vectorized sign kernel for float16 using half2
__global__ void sign_kernel_half_vec2(const __half* __restrict__ x, 
                                       __half* __restrict__ output, 
                                       int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int vec_size = size / 2;
    
    const __half zero = __float2half(0.0f);
    const __half one = __float2half(1.0f);
    const __half neg_one = __float2half(-1.0f);
    
    // Vectorized processing using half2
    if (idx < vec_size) {
        half2 val = reinterpret_cast<const half2*>(x)[idx];
        half2 result;
        
        float val_x = __half2float(val.x);
        float val_y = __half2float(val.y);
        
        result.x = (val_x == 0.0f) ? zero : ((val_x > 0.0f) ? one : neg_one);
        result.y = (val_y == 0.0f) ? zero : ((val_y > 0.0f) ? one : neg_one);
        
        reinterpret_cast<half2*>(output)[idx] = result;
    }
    
    // Handle remaining element
    if (size % 2 == 1 && idx == 0) {
        int last_idx = size - 1;
        __half val = x[last_idx];
        float val_f = __half2float(val);
        output[last_idx] = (val_f == 0.0f) ? zero : ((val_f > 0.0f) ? one : neg_one);
    }
}

extern "C" {

// Main entry point expected by the test framework
// dtype_id: 6 for float32, 5 for float16 (PyTorch ScalarType enum)
void cuda_kernel(const void* x, void* output, int size, int dtype_id) {
    // dtype_id from PyTorch: 6 = float32, 5 = float16
    if (dtype_id == 5) {
        // Float16 path
        bool is_aligned = (reinterpret_cast<std::uintptr_t>(x) % 4 == 0) && 
                          (reinterpret_cast<std::uintptr_t>(output) % 4 == 0) &&
                          (size % 2 == 0);
        
        if (is_aligned && size >= 512) {
            // Use vectorized kernel for larger aligned data
            int threads = 256;
            int vec_size = size / 2;
            int blocks = (vec_size + threads - 1) / threads;
            blocks = min(blocks, 1024);
            
            sign_kernel_half_vec2<<<blocks, threads>>>(
                reinterpret_cast<const __half*>(x), 
                reinterpret_cast<__half*>(output), 
                size
            );
        } else {
            // Use standard kernel
            int threads = 256;
            int blocks = (size + threads - 1) / threads;
            blocks = min(blocks, 2048);
            
            sign_kernel_half<<<blocks, threads>>>(
                reinterpret_cast<const __half*>(x), 
                reinterpret_cast<__half*>(output), 
                size
            );
        }
    } else {
        // Float32 path (default)
        bool is_aligned = (reinterpret_cast<std::uintptr_t>(x) % 16 == 0) && 
                          (reinterpret_cast<std::uintptr_t>(output) % 16 == 0) &&
                          (size % 4 == 0);
        
        if (is_aligned && size >= 1024) {
            // Use vectorized kernel for larger aligned data
            int threads = 256;
            int vec_size = size / 4;
            int blocks = (vec_size + threads - 1) / threads;
            blocks = min(blocks, 1024);
            
            sign_kernel_float_vec4<<<blocks, threads>>>(
                reinterpret_cast<const float*>(x), 
                reinterpret_cast<float*>(output), 
                size
            );
        } else {
            // Use standard kernel
            int threads = 256;
            int blocks = (size + threads - 1) / threads;
            blocks = min(blocks, 2048);
            
            sign_kernel_float<<<blocks, threads>>>(
                reinterpret_cast<const float*>(x), 
                reinterpret_cast<float*>(output), 
                size
            );
        }
    }
}

} // extern "C"
