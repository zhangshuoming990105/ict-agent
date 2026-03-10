#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>

// Swish activation: x * sigmoid(x) = x / (1 + exp(-x))
// More stable form: x * (1 / (1 + exp(-x)))

// ========== Float32 kernels ==========

__device__ __forceinline__ float swish_f32(float x) {
    return x / (1.0f + expf(-x));
}

// OP_TYPE 0: Float32 scalar
__global__ void swish_f32_scalar(const float* __restrict__ input,
                                  float* __restrict__ output,
                                  int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    for (int i = idx; i < N; i += stride) {
        output[i] = swish_f32(input[i]);
    }
}

// OP_TYPE 1: Float32 vectorized (float4)
__global__ void swish_f32_vec4(const float* __restrict__ input,
                                float* __restrict__ output,
                                int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    const int vec_size = 4;
    int N_vec = N / vec_size;
    
    const float4* input_vec = reinterpret_cast<const float4*>(input);
    float4* output_vec = reinterpret_cast<float4*>(output);
    
    for (int i = idx; i < N_vec; i += stride) {
        float4 val = input_vec[i];
        float4 result;
        result.x = swish_f32(val.x);
        result.y = swish_f32(val.y);
        result.z = swish_f32(val.z);
        result.w = swish_f32(val.w);
        output_vec[i] = result;
    }
    
    // Handle remainder
    int remainder_start = N_vec * vec_size + idx;
    if (remainder_start < N) {
        for (int i = remainder_start; i < N; i += stride) {
            output[i] = swish_f32(input[i]);
        }
    }
}

// ========== Float16 kernels ==========

__device__ __forceinline__ half swish_f16(half x) {
    float x_f32 = __half2float(x);
    float result = x_f32 / (1.0f + expf(-x_f32));
    return __float2half(result);
}

__device__ __forceinline__ half2 swish_f16x2(half2 x) {
    float2 x_f32 = __half22float2(x);
    float2 result;
    result.x = x_f32.x / (1.0f + expf(-x_f32.x));
    result.y = x_f32.y / (1.0f + expf(-x_f32.y));
    return __float22half2_rn(result);
}

// OP_TYPE 2: Float16 scalar
__global__ void swish_f16_scalar(const half* __restrict__ input,
                                  half* __restrict__ output,
                                  int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    for (int i = idx; i < N; i += stride) {
        output[i] = swish_f16(input[i]);
    }
}

// OP_TYPE 3: Float16 vectorized (half2)
__global__ void swish_f16_vec2(const half* __restrict__ input,
                                half* __restrict__ output,
                                int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    const int vec_size = 2;
    int N_vec = N / vec_size;
    
    const half2* input_vec = reinterpret_cast<const half2*>(input);
    half2* output_vec = reinterpret_cast<half2*>(output);
    
    for (int i = idx; i < N_vec; i += stride) {
        half2 val = input_vec[i];
        output_vec[i] = swish_f16x2(val);
    }
    
    // Handle remainder
    int remainder_start = N_vec * vec_size + idx;
    if (remainder_start < N) {
        for (int i = remainder_start; i < N; i += stride) {
            output[i] = swish_f16(input[i]);
        }
    }
}

// OP_TYPE 4: Float16 vectorized (8 elements, unpack)
__global__ void swish_f16_vec8_unpack(const half* __restrict__ input,
                                       half* __restrict__ output,
                                       int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    const int vec_size = 8;
    int N_vec = N / vec_size;
    
    for (int i = idx; i < N_vec; i += stride) {
        int base_idx = i * vec_size;
        
        // Load 4 half2 vectors (8 half elements)
        half2 v0 = *reinterpret_cast<const half2*>(&input[base_idx + 0]);
        half2 v1 = *reinterpret_cast<const half2*>(&input[base_idx + 2]);
        half2 v2 = *reinterpret_cast<const half2*>(&input[base_idx + 4]);
        half2 v3 = *reinterpret_cast<const half2*>(&input[base_idx + 6]);
        
        // Process
        half2 r0 = swish_f16x2(v0);
        half2 r1 = swish_f16x2(v1);
        half2 r2 = swish_f16x2(v2);
        half2 r3 = swish_f16x2(v3);
        
        // Store
        *reinterpret_cast<half2*>(&output[base_idx + 0]) = r0;
        *reinterpret_cast<half2*>(&output[base_idx + 2]) = r1;
        *reinterpret_cast<half2*>(&output[base_idx + 4]) = r2;
        *reinterpret_cast<half2*>(&output[base_idx + 6]) = r3;
    }
    
    // Handle remainder
    int remainder_start = N_vec * vec_size + idx;
    if (remainder_start < N) {
        for (int i = remainder_start; i < N; i += stride) {
            output[i] = swish_f16(input[i]);
        }
    }
}

// OP_TYPE 5: Float16 vectorized (8 elements, pack with float4)
__global__ void swish_f16_vec8_pack(const half* __restrict__ input,
                                     half* __restrict__ output,
                                     int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    const int vec_size = 8;
    int N_vec = N / vec_size;
    
    // Use float4 to load/store 8 half elements at once
    const float4* input_vec = reinterpret_cast<const float4*>(input);
    float4* output_vec = reinterpret_cast<float4*>(output);
    
    for (int i = idx; i < N_vec; i += stride) {
        float4 val = input_vec[i];
        
        // Reinterpret as half2 vectors
        half2* val_h2 = reinterpret_cast<half2*>(&val);
        
        half2 r0 = swish_f16x2(val_h2[0]);
        half2 r1 = swish_f16x2(val_h2[1]);
        half2 r2 = swish_f16x2(val_h2[2]);
        half2 r3 = swish_f16x2(val_h2[3]);
        
        // Pack results back into float4
        float4 result;
        half2* result_h2 = reinterpret_cast<half2*>(&result);
        result_h2[0] = r0;
        result_h2[1] = r1;
        result_h2[2] = r2;
        result_h2[3] = r3;
        
        output_vec[i] = result;
    }
    
    // Handle remainder
    int remainder_start = N_vec * vec_size + idx;
    if (remainder_start < N) {
        for (int i = remainder_start; i < N; i += stride) {
            output[i] = swish_f16(input[i]);
        }
    }
}

// ========== Dispatcher ==========

extern "C" void cuda_kernel(void* input, void* output, int N, int op_type) {
    const int threads = 256;
    const int blocks = (N + threads - 1) / threads;
    const int max_blocks = 1024;
    const int final_blocks = min(blocks, max_blocks);
    
    switch(op_type) {
        case 0: // f32 scalar
            swish_f32_scalar<<<final_blocks, threads>>>(
                static_cast<const float*>(input),
                static_cast<float*>(output),
                N
            );
            break;
        case 1: // f32x4
            swish_f32_vec4<<<final_blocks, threads>>>(
                static_cast<const float*>(input),
                static_cast<float*>(output),
                N
            );
            break;
        case 2: // f16 scalar
            swish_f16_scalar<<<final_blocks, threads>>>(
                static_cast<const half*>(input),
                static_cast<half*>(output),
                N
            );
            break;
        case 3: // f16x2
            swish_f16_vec2<<<final_blocks, threads>>>(
                static_cast<const half*>(input),
                static_cast<half*>(output),
                N
            );
            break;
        case 4: // f16x8 unpack
            swish_f16_vec8_unpack<<<final_blocks, threads>>>(
                static_cast<const half*>(input),
                static_cast<half*>(output),
                N
            );
            break;
        case 5: // f16x8 pack
            swish_f16_vec8_pack<<<final_blocks, threads>>>(
                static_cast<const half*>(input),
                static_cast<half*>(output),
                N
            );
            break;
        default:
            // Fallback to scalar
            swish_f32_scalar<<<final_blocks, threads>>>(
                static_cast<const float*>(input),
                static_cast<float*>(output),
                N
            );
            break;
    }
}
