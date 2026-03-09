#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>

// OP_TYPE mapping:
// 0: float32 scalar
// 1: float32 vectorized (float4)
// 2: float16 scalar
// 3: float16 vectorized (half2)
// 4: float16 vectorized (8 elements with unpack)
// 5: float16 vectorized (8 elements with packed ops)

// ============================================================================
// Float32 Kernels
// ============================================================================

// FP32 Scalar version
__global__ void sigmoid_f32_scalar(const float* __restrict__ input,
                                     float* __restrict__ output,
                                     int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    for (int i = idx; i < N; i += stride) {
        float x = input[i];
        output[i] = 1.0f / (1.0f + expf(-x));
    }
}

// FP32 Vectorized version (float4)
__global__ void sigmoid_f32_vec4(const float* __restrict__ input,
                                   float* __restrict__ output,
                                   int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int vec_size = N / 4;
    
    const float4* input_vec = reinterpret_cast<const float4*>(input);
    float4* output_vec = reinterpret_cast<float4*>(output);
    
    for (int i = idx; i < vec_size; i += stride) {
        float4 val = input_vec[i];
        val.x = 1.0f / (1.0f + expf(-val.x));
        val.y = 1.0f / (1.0f + expf(-val.y));
        val.z = 1.0f / (1.0f + expf(-val.z));
        val.w = 1.0f / (1.0f + expf(-val.w));
        output_vec[i] = val;
    }
    
    // Handle remaining elements
    int remaining_start = vec_size * 4;
    for (int i = remaining_start + idx; i < N; i += stride) {
        float x = input[i];
        output[i] = 1.0f / (1.0f + expf(-x));
    }
}

// ============================================================================
// Float16 Kernels
// ============================================================================

// FP16 Scalar version
__global__ void sigmoid_f16_scalar(const __half* __restrict__ input,
                                     __half* __restrict__ output,
                                     int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    for (int i = idx; i < N; i += stride) {
        float x = __half2float(input[i]);
        output[i] = __float2half(1.0f / (1.0f + expf(-x)));
    }
}

// FP16 Vectorized version (half2)
__global__ void sigmoid_f16_vec2(const __half* __restrict__ input,
                                   __half* __restrict__ output,
                                   int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int vec_size = N / 2;
    
    const half2* input_vec = reinterpret_cast<const half2*>(input);
    half2* output_vec = reinterpret_cast<half2*>(output);
    
    for (int i = idx; i < vec_size; i += stride) {
        half2 val = input_vec[i];
        float x = __half2float(val.x);
        float y = __half2float(val.y);
        val.x = __float2half(1.0f / (1.0f + expf(-x)));
        val.y = __float2half(1.0f / (1.0f + expf(-y)));
        output_vec[i] = val;
    }
    
    // Handle remaining element (if N is odd)
    int remaining_start = vec_size * 2;
    if (remaining_start < N && idx == 0) {
        float x = __half2float(input[remaining_start]);
        output[remaining_start] = __float2half(1.0f / (1.0f + expf(-x)));
    }
}

// FP16 Vectorized version (8 elements with unpack)
__global__ void sigmoid_f16_vec8_unpack(const __half* __restrict__ input,
                                          __half* __restrict__ output,
                                          int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int vec_size = N / 8;
    
    for (int i = idx; i < vec_size; i += stride) {
        int base_idx = i * 8;
        
        // Load 8 half values
        half2 v0 = *reinterpret_cast<const half2*>(&input[base_idx + 0]);
        half2 v1 = *reinterpret_cast<const half2*>(&input[base_idx + 2]);
        half2 v2 = *reinterpret_cast<const half2*>(&input[base_idx + 4]);
        half2 v3 = *reinterpret_cast<const half2*>(&input[base_idx + 6]);
        
        // Process each element
        float x0 = __half2float(v0.x);
        float x1 = __half2float(v0.y);
        float x2 = __half2float(v1.x);
        float x3 = __half2float(v1.y);
        float x4 = __half2float(v2.x);
        float x5 = __half2float(v2.y);
        float x6 = __half2float(v3.x);
        float x7 = __half2float(v3.y);
        
        v0.x = __float2half(1.0f / (1.0f + expf(-x0)));
        v0.y = __float2half(1.0f / (1.0f + expf(-x1)));
        v1.x = __float2half(1.0f / (1.0f + expf(-x2)));
        v1.y = __float2half(1.0f / (1.0f + expf(-x3)));
        v2.x = __float2half(1.0f / (1.0f + expf(-x4)));
        v2.y = __float2half(1.0f / (1.0f + expf(-x5)));
        v3.x = __float2half(1.0f / (1.0f + expf(-x6)));
        v3.y = __float2half(1.0f / (1.0f + expf(-x7)));
        
        // Store results
        *reinterpret_cast<half2*>(&output[base_idx + 0]) = v0;
        *reinterpret_cast<half2*>(&output[base_idx + 2]) = v1;
        *reinterpret_cast<half2*>(&output[base_idx + 4]) = v2;
        *reinterpret_cast<half2*>(&output[base_idx + 6]) = v3;
    }
    
    // Handle remaining elements
    int remaining_start = vec_size * 8;
    for (int i = remaining_start + idx; i < N; i += stride) {
        float x = __half2float(input[i]);
        output[i] = __float2half(1.0f / (1.0f + expf(-x)));
    }
}

// FP16 Vectorized version (8 elements with packed half2 ops)
__global__ void sigmoid_f16_vec8_pack(const __half* __restrict__ input,
                                        __half* __restrict__ output,
                                        int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int vec_size = N / 8;
    
    for (int i = idx; i < vec_size; i += stride) {
        int base_idx = i * 8;
        
        // Load 8 half values as 4 half2
        half2 v0 = *reinterpret_cast<const half2*>(&input[base_idx + 0]);
        half2 v1 = *reinterpret_cast<const half2*>(&input[base_idx + 2]);
        half2 v2 = *reinterpret_cast<const half2*>(&input[base_idx + 4]);
        half2 v3 = *reinterpret_cast<const half2*>(&input[base_idx + 6]);
        
        // Compute sigmoid using half2 operations
        // For half2, we need to convert to float for exp
        float2 f0 = __half22float2(v0);
        float2 f1 = __half22float2(v1);
        float2 f2 = __half22float2(v2);
        float2 f3 = __half22float2(v3);
        
        f0.x = 1.0f / (1.0f + expf(-f0.x));
        f0.y = 1.0f / (1.0f + expf(-f0.y));
        f1.x = 1.0f / (1.0f + expf(-f1.x));
        f1.y = 1.0f / (1.0f + expf(-f1.y));
        f2.x = 1.0f / (1.0f + expf(-f2.x));
        f2.y = 1.0f / (1.0f + expf(-f2.y));
        f3.x = 1.0f / (1.0f + expf(-f3.x));
        f3.y = 1.0f / (1.0f + expf(-f3.y));
        
        v0 = __float22half2_rn(f0);
        v1 = __float22half2_rn(f1);
        v2 = __float22half2_rn(f2);
        v3 = __float22half2_rn(f3);
        
        // Store results
        *reinterpret_cast<half2*>(&output[base_idx + 0]) = v0;
        *reinterpret_cast<half2*>(&output[base_idx + 2]) = v1;
        *reinterpret_cast<half2*>(&output[base_idx + 4]) = v2;
        *reinterpret_cast<half2*>(&output[base_idx + 6]) = v3;
    }
    
    // Handle remaining elements
    int remaining_start = vec_size * 8;
    for (int i = remaining_start + idx; i < N; i += stride) {
        float x = __half2float(input[i]);
        output[i] = __float2half(1.0f / (1.0f + expf(-x)));
    }
}

// ============================================================================
// Entry Point
// ============================================================================

extern "C" void cuda_kernel(const void* input, void* output, int N, int op_type) {
    const int threads_per_block = 256;
    const int blocks = min((N + threads_per_block - 1) / threads_per_block, 1024);
    
    switch (op_type) {
        case 0: // FP32 scalar
            sigmoid_f32_scalar<<<blocks, threads_per_block>>>(
                static_cast<const float*>(input),
                static_cast<float*>(output),
                N
            );
            break;
            
        case 1: // FP32 vectorized (float4)
            sigmoid_f32_vec4<<<blocks, threads_per_block>>>(
                static_cast<const float*>(input),
                static_cast<float*>(output),
                N
            );
            break;
            
        case 2: // FP16 scalar
            sigmoid_f16_scalar<<<blocks, threads_per_block>>>(
                static_cast<const __half*>(input),
                static_cast<__half*>(output),
                N
            );
            break;
            
        case 3: // FP16 vectorized (half2)
            sigmoid_f16_vec2<<<blocks, threads_per_block>>>(
                static_cast<const __half*>(input),
                static_cast<__half*>(output),
                N
            );
            break;
            
        case 4: // FP16 vectorized (8 elements with unpack)
            sigmoid_f16_vec8_unpack<<<blocks, threads_per_block>>>(
                static_cast<const __half*>(input),
                static_cast<__half*>(output),
                N
            );
            break;
            
        case 5: // FP16 vectorized (8 elements with pack)
            sigmoid_f16_vec8_pack<<<blocks, threads_per_block>>>(
                static_cast<const __half*>(input),
                static_cast<__half*>(output),
                N
            );
            break;
            
        default:
            // Fallback to FP32 scalar
            sigmoid_f32_scalar<<<blocks, threads_per_block>>>(
                static_cast<const float*>(input),
                static_cast<float*>(output),
                N
            );
            break;
    }
}
