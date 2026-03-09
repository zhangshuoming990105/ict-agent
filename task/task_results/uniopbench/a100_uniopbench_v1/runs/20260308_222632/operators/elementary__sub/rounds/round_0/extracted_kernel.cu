#include <cuda_runtime.h>
#include <cuda_fp16.h>

// Main kernel template for generic types
template<typename T>
__global__ void sub_kernel(const T* a, const T* b, T* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    // Grid-stride loop
    for (int i = idx; i < size; i += stride) {
        output[i] = a[i] - b[i];
    }
}

// Optimized kernel for float32 with vectorized loads (float4 = 4 floats)
__global__ void sub_kernel_float_vectorized(const float* a, const float* b, float* output, int size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    // Process 4 elements at a time when possible
    int vec_size = size / 4;
    int remainder = size % 4;
    
    // Vectorized processing
    for (int i = tid; i < vec_size; i += stride) {
        int idx = i * 4;
        float4 a_vec = *reinterpret_cast<const float4*>(&a[idx]);
        float4 b_vec = *reinterpret_cast<const float4*>(&b[idx]);
        float4 out_vec;
        out_vec.x = a_vec.x - b_vec.x;
        out_vec.y = a_vec.y - b_vec.y;
        out_vec.z = a_vec.z - b_vec.z;
        out_vec.w = a_vec.w - b_vec.w;
        *reinterpret_cast<float4*>(&output[idx]) = out_vec;
    }
    
    // Handle remainder elements
    int remainder_start = vec_size * 4;
    for (int i = tid; i < remainder; i += stride) {
        int idx = remainder_start + i;
        if (idx < size) {
            output[idx] = a[idx] - b[idx];
        }
    }
}

// Optimized kernel for float16 with vectorized loads using half2
__global__ void sub_kernel_half_vectorized(const __half* a, const __half* b, __half* output, int size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    // Process 8 elements at a time (float4 = 8 half precision values = 16 bytes)
    int vec_size = size / 8;
    int remainder = size % 8;
    
    // Vectorized processing using float4 for 128-bit loads
    for (int i = tid; i < vec_size; i += stride) {
        int idx = i * 8;
        float4 a_vec = *reinterpret_cast<const float4*>(&a[idx]);
        float4 b_vec = *reinterpret_cast<const float4*>(&b[idx]);
        
        // Process as half2 pairs for proper arithmetic
        __half2* a_h2 = reinterpret_cast<__half2*>(&a_vec);
        __half2* b_h2 = reinterpret_cast<__half2*>(&b_vec);
        float4 out_vec;
        __half2* out_h2 = reinterpret_cast<__half2*>(&out_vec);
        
        out_h2[0] = __hsub2(a_h2[0], b_h2[0]);
        out_h2[1] = __hsub2(a_h2[1], b_h2[1]);
        out_h2[2] = __hsub2(a_h2[2], b_h2[2]);
        out_h2[3] = __hsub2(a_h2[3], b_h2[3]);
        
        *reinterpret_cast<float4*>(&output[idx]) = out_vec;
    }
    
    // Handle remainder elements
    int remainder_start = vec_size * 8;
    for (int i = tid; i < remainder; i += stride) {
        int idx = remainder_start + i;
        if (idx < size) {
            output[idx] = __hsub(a[idx], b[idx]);
        }
    }
}

// C interface - the main entry point expected by the framework
extern "C" {

// Main cuda_kernel function that the framework will call
// dtype_id: 0 = float32, 1 = float16
void cuda_kernel(const void* a, const void* b, void* output, int size, int dtype_id) {
    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;
    
    if (dtype_id == 0) {
        // Float32 path
        if (size >= 1024 && size % 4 == 0) {
            sub_kernel_float_vectorized<<<num_blocks, block_size>>>(
                static_cast<const float*>(a),
                static_cast<const float*>(b),
                static_cast<float*>(output),
                size
            );
        } else {
            sub_kernel<float><<<num_blocks, block_size>>>(
                static_cast<const float*>(a),
                static_cast<const float*>(b),
                static_cast<float*>(output),
                size
            );
        }
    } else if (dtype_id == 1) {
        // Float16 path
        if (size >= 1024 && size % 8 == 0) {
            sub_kernel_half_vectorized<<<num_blocks, block_size>>>(
                static_cast<const __half*>(a),
                static_cast<const __half*>(b),
                static_cast<__half*>(output),
                size
            );
        } else {
            sub_kernel<__half><<<num_blocks, block_size>>>(
                static_cast<const __half*>(a),
                static_cast<const __half*>(b),
                static_cast<__half*>(output),
                size
            );
        }
    }
}

} // extern "C"
