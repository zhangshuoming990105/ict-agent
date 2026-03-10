#include <cuda_runtime.h>
#include <cuda_fp16.h>

// ============================================================================
// ReLU CUDA Kernel Implementation
// Supports multiple optimization levels for FP32 and FP16
// ============================================================================

// FP32 Scalar ReLU (OP_TYPE = 0)
__global__ void relu_f32_scalar_kernel(const float* __restrict__ input,
                                        float* __restrict__ output,
                                        int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        output[idx] = fmaxf(input[idx], 0.0f);
    }
}

// FP32 Vectorized ReLU using float4 (OP_TYPE = 1)
__global__ void relu_f32_vec4_kernel(const float4* __restrict__ input,
                                      float4* __restrict__ output,
                                      int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int vec_count = N / 4;
    
    if (idx < vec_count) {
        float4 val = input[idx];
        val.x = fmaxf(val.x, 0.0f);
        val.y = fmaxf(val.y, 0.0f);
        val.z = fmaxf(val.z, 0.0f);
        val.w = fmaxf(val.w, 0.0f);
        output[idx] = val;
    }
}

// FP16 Scalar ReLU (OP_TYPE = 2)
__global__ void relu_f16_scalar_kernel(const __half* __restrict__ input,
                                        __half* __restrict__ output,
                                        int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        __half val = input[idx];
        output[idx] = __hgt(val, __float2half(0.0f)) ? val : __float2half(0.0f);
    }
}

// FP16 Vectorized ReLU using half2 (OP_TYPE = 3)
__global__ void relu_f16_vec2_kernel(const half2* __restrict__ input,
                                      half2* __restrict__ output,
                                      int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int vec_count = N / 2;
    
    if (idx < vec_count) {
        half2 val = input[idx];
        half2 zero = __float2half2_rn(0.0f);
        output[idx] = __hmax2(val, zero);
    }
}

// FP16 Vectorized ReLU using 8 elements with unpack (OP_TYPE = 4)
__global__ void relu_f16_vec8_unpack_kernel(const __half* __restrict__ input,
                                             __half* __restrict__ output,
                                             int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int base_idx = idx * 8;
    
    if (base_idx + 7 < N) {
        // Load 8 half values
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            __half val = input[base_idx + i];
            output[base_idx + i] = __hgt(val, __float2half(0.0f)) ? val : __float2half(0.0f);
        }
    }
}

// FP16 Vectorized ReLU using 8 elements with packed half2 (OP_TYPE = 5)
__global__ void relu_f16_vec8_pack_kernel(const half2* __restrict__ input,
                                           half2* __restrict__ output,
                                           int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int base_idx = idx * 4;  // 4 half2 = 8 halves
    int vec_count = N / 8;
    
    if (idx < vec_count) {
        half2 zero = __float2half2_rn(0.0f);
        
        // Process 4 half2 values (8 halves total)
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            half2 val = input[base_idx + i];
            output[base_idx + i] = __hmax2(val, zero);
        }
    }
}

// ============================================================================
// Host function dispatcher
// ============================================================================

extern "C" {

void cuda_kernel(const void* input, void* output, int N, int op_type) {
    // Launch configuration
    const int threads = 256;
    int blocks;
    
    switch (op_type) {
        case 0: // FP32 scalar
            blocks = (N + threads - 1) / threads;
            relu_f32_scalar_kernel<<<blocks, threads>>>(
                reinterpret_cast<const float*>(input),
                reinterpret_cast<float*>(output),
                N
            );
            break;
            
        case 1: // FP32 vec4
            blocks = ((N / 4) + threads - 1) / threads;
            relu_f32_vec4_kernel<<<blocks, threads>>>(
                reinterpret_cast<const float4*>(input),
                reinterpret_cast<float4*>(output),
                N
            );
            break;
            
        case 2: // FP16 scalar
            blocks = (N + threads - 1) / threads;
            relu_f16_scalar_kernel<<<blocks, threads>>>(
                reinterpret_cast<const __half*>(input),
                reinterpret_cast<__half*>(output),
                N
            );
            break;
            
        case 3: // FP16 vec2
            blocks = ((N / 2) + threads - 1) / threads;
            relu_f16_vec2_kernel<<<blocks, threads>>>(
                reinterpret_cast<const half2*>(input),
                reinterpret_cast<half2*>(output),
                N
            );
            break;
            
        case 4: // FP16 vec8 unpack
            blocks = ((N / 8) + threads - 1) / threads;
            relu_f16_vec8_unpack_kernel<<<blocks, threads>>>(
                reinterpret_cast<const __half*>(input),
                reinterpret_cast<__half*>(output),
                N
            );
            break;
            
        case 5: // FP16 vec8 pack
            blocks = ((N / 8) + threads - 1) / threads;
            relu_f16_vec8_pack_kernel<<<blocks, threads>>>(
                reinterpret_cast<const half2*>(input),
                reinterpret_cast<half2*>(output),
                N
            );
            break;
            
        default:
            // Fallback to scalar for unknown op_type
            blocks = (N + threads - 1) / threads;
            relu_f32_scalar_kernel<<<blocks, threads>>>(
                reinterpret_cast<const float*>(input),
                reinterpret_cast<float*>(output),
                N
            );
            break;
    }
}

} // extern "C"
