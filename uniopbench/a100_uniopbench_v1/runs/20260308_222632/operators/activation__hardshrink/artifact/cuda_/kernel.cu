#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

// Hardshrink operation: output = (|x| > 0.5) ? x : 0
// lambd = 0.5

// ============================================================================
// Float32 Operations
// ============================================================================

__device__ __forceinline__ float hardshrink_f32(float x) {
    float abs_x = fabsf(x);
    return (abs_x > 0.5f) ? x : 0.0f;
}

__device__ __forceinline__ float4 hardshrink_f32x4(float4 x) {
    float4 result;
    result.x = hardshrink_f32(x.x);
    result.y = hardshrink_f32(x.y);
    result.z = hardshrink_f32(x.z);
    result.w = hardshrink_f32(x.w);
    return result;
}

// ============================================================================
// Float16 Operations
// ============================================================================

__device__ __forceinline__ half hardshrink_f16(half x) {
    float abs_x = fabsf(__half2float(x));
    return (abs_x > 0.5f) ? x : __float2half(0.0f);
}

__device__ __forceinline__ half2 hardshrink_f16x2(half2 x) {
    half2 result;
    result.x = hardshrink_f16(x.x);
    result.y = hardshrink_f16(x.y);
    return result;
}

// Float16x8 using unpacked operations
struct alignas(16) half8_t {
    half data[8];
};

__device__ __forceinline__ half8_t hardshrink_f16x8_unpack(half8_t x) {
    half8_t result;
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        result.data[i] = hardshrink_f16(x.data[i]);
    }
    return result;
}

// Float16x8 using packed half2 operations
__device__ __forceinline__ half8_t hardshrink_f16x8_pack(half8_t x) {
    half8_t result;
    half2* x_h2 = reinterpret_cast<half2*>(&x);
    half2* result_h2 = reinterpret_cast<half2*>(&result);
    
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        result_h2[i] = hardshrink_f16x2(x_h2[i]);
    }
    return result;
}

// ============================================================================
// Kernel Implementations
// ============================================================================

template<typename T, int OP_TYPE>
__global__ void hardshrink_kernel_impl(const T* input, T* output, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    if (OP_TYPE == 0) {
        // Float32 scalar
        for (int i = idx; i < N; i += stride) {
            output[i] = hardshrink_f32(input[i]);
        }
    } else if (OP_TYPE == 1) {
        // Float32x4 vectorized
        const float4* input_vec = reinterpret_cast<const float4*>(input);
        float4* output_vec = reinterpret_cast<float4*>(output);
        int vec_size = N / 4;
        
        for (int i = idx; i < vec_size; i += stride) {
            output_vec[i] = hardshrink_f32x4(input_vec[i]);
        }
    } else if (OP_TYPE == 2) {
        // Float16 scalar
        for (int i = idx; i < N; i += stride) {
            output[i] = hardshrink_f16(input[i]);
        }
    } else if (OP_TYPE == 3) {
        // Float16x2 vectorized
        const half2* input_vec = reinterpret_cast<const half2*>(input);
        half2* output_vec = reinterpret_cast<half2*>(output);
        int vec_size = N / 2;
        
        for (int i = idx; i < vec_size; i += stride) {
            output_vec[i] = hardshrink_f16x2(input_vec[i]);
        }
    } else if (OP_TYPE == 4) {
        // Float16x8 unpack
        const half8_t* input_vec = reinterpret_cast<const half8_t*>(input);
        half8_t* output_vec = reinterpret_cast<half8_t*>(output);
        int vec_size = N / 8;
        
        for (int i = idx; i < vec_size; i += stride) {
            output_vec[i] = hardshrink_f16x8_unpack(input_vec[i]);
        }
    } else if (OP_TYPE == 5) {
        // Float16x8 pack
        const half8_t* input_vec = reinterpret_cast<const half8_t*>(input);
        half8_t* output_vec = reinterpret_cast<half8_t*>(output);
        int vec_size = N / 8;
        
        for (int i = idx; i < vec_size; i += stride) {
            output_vec[i] = hardshrink_f16x8_pack(input_vec[i]);
        }
    }
}

// ============================================================================
// Host API - Expected name is "cuda_kernel"
// ============================================================================

extern "C" {

void cuda_kernel(const void* input, void* output, int N, int op_type) {
    const int threads = 256;
    const int blocks = min((N + threads - 1) / threads, 1024);
    
    switch(op_type) {
        case 0: // Float32 scalar
            hardshrink_kernel_impl<float, 0><<<blocks, threads>>>(
                static_cast<const float*>(input),
                static_cast<float*>(output),
                N
            );
            break;
        case 1: // Float32x4
            hardshrink_kernel_impl<float, 1><<<blocks, threads>>>(
                static_cast<const float*>(input),
                static_cast<float*>(output),
                N
            );
            break;
        case 2: // Float16 scalar
            hardshrink_kernel_impl<half, 2><<<blocks, threads>>>(
                static_cast<const half*>(input),
                static_cast<half*>(output),
                N
            );
            break;
        case 3: // Float16x2
            hardshrink_kernel_impl<half, 3><<<blocks, threads>>>(
                static_cast<const half*>(input),
                static_cast<half*>(output),
                N
            );
            break;
        case 4: // Float16x8 unpack
            hardshrink_kernel_impl<half, 4><<<blocks, threads>>>(
                static_cast<const half*>(input),
                static_cast<half*>(output),
                N
            );
            break;
        case 5: // Float16x8 pack
            hardshrink_kernel_impl<half, 5><<<blocks, threads>>>(
                static_cast<const half*>(input),
                static_cast<half*>(output),
                N
            );
            break;
        default:
            // Invalid op_type, do nothing
            break;
    }
}

} // extern "C"
