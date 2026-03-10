#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>

// ELU activation: elu(x) = x if x > 0, else exp(x) - 1 (alpha=1.0)

// Helper for FP16 operations
__device__ __forceinline__ half elu_fp16(half x) {
    #if __CUDA_ARCH__ >= 530
    if (__hgt(x, __float2half(0.0f))) {
        return x;
    } else {
        return __hsub(hexp(x), __float2half(1.0f));
    }
    #else
    float fx = __half2float(x);
    return __float2half(fx > 0.0f ? fx : expf(fx) - 1.0f);
    #endif
}

__device__ __forceinline__ float elu_fp32(float x) {
    return x > 0.0f ? x : expf(x) - 1.0f;
}

// ============================================================================
// FP32 Kernels
// ============================================================================

// Op Type 0: FP32 Scalar
__global__ void elu_fp32_scalar(const float* __restrict__ input,
                                 float* __restrict__ output,
                                 int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        output[idx] = elu_fp32(input[idx]);
    }
}

// Op Type 1: FP32 Vec4
__global__ void elu_fp32_vec4(const float* __restrict__ input,
                               float* __restrict__ output,
                               int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int vec_idx = idx * 4;
    
    if (vec_idx + 3 < N) {
        float4 in = reinterpret_cast<const float4*>(input)[idx];
        float4 out;
        out.x = elu_fp32(in.x);
        out.y = elu_fp32(in.y);
        out.z = elu_fp32(in.z);
        out.w = elu_fp32(in.w);
        reinterpret_cast<float4*>(output)[idx] = out;
    } else if (vec_idx < N) {
        // Handle tail elements
        for (int i = vec_idx; i < N; i++) {
            output[i] = elu_fp32(input[i]);
        }
    }
}

// ============================================================================
// FP16 Kernels
// ============================================================================

// Op Type 2: FP16 Scalar
__global__ void elu_fp16_scalar(const half* __restrict__ input,
                                 half* __restrict__ output,
                                 int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        output[idx] = elu_fp16(input[idx]);
    }
}

// Op Type 3: FP16 Vec2 (half2)
__global__ void elu_fp16_vec2(const half* __restrict__ input,
                               half* __restrict__ output,
                               int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int vec_idx = idx * 2;
    
    if (vec_idx + 1 < N) {
        half2 in = reinterpret_cast<const half2*>(input)[idx];
        half2 out;
        out.x = elu_fp16(in.x);
        out.y = elu_fp16(in.y);
        reinterpret_cast<half2*>(output)[idx] = out;
    } else if (vec_idx < N) {
        output[vec_idx] = elu_fp16(input[vec_idx]);
    }
}

// Op Type 4: FP16 Vec8 (using float4 for memory access)
__global__ void elu_fp16_vec8_unpack(const half* __restrict__ input,
                                      half* __restrict__ output,
                                      int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int vec_idx = idx * 8;
    
    if (vec_idx + 7 < N) {
        // Load 8 half values as float4 (128-bit aligned access)
        float4 in = reinterpret_cast<const float4*>(input)[idx];
        half* in_half = reinterpret_cast<half*>(&in);
        
        half out_half[8];
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            out_half[i] = elu_fp16(in_half[i]);
        }
        
        // Store 8 half values as float4
        float4 out;
        half* out_ptr = reinterpret_cast<half*>(&out);
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            out_ptr[i] = out_half[i];
        }
        reinterpret_cast<float4*>(output)[idx] = out;
    } else if (vec_idx < N) {
        // Handle tail elements
        for (int i = vec_idx; i < N; i++) {
            output[i] = elu_fp16(input[i]);
        }
    }
}

// Op Type 5: FP16 Vec8 with half2 packed operations
__global__ void elu_fp16_vec8_pack(const half* __restrict__ input,
                                    half* __restrict__ output,
                                    int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int vec_idx = idx * 8;
    
    if (vec_idx + 7 < N) {
        // Load 8 half values as float4
        float4 in = reinterpret_cast<const float4*>(input)[idx];
        half2* in_half2 = reinterpret_cast<half2*>(&in);
        
        half2 out_half2[4];
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            out_half2[i].x = elu_fp16(in_half2[i].x);
            out_half2[i].y = elu_fp16(in_half2[i].y);
        }
        
        // Store back
        float4 out;
        half2* out_ptr = reinterpret_cast<half2*>(&out);
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            out_ptr[i] = out_half2[i];
        }
        reinterpret_cast<float4*>(output)[idx] = out;
    } else if (vec_idx < N) {
        // Handle tail elements
        for (int i = vec_idx; i < N; i++) {
            output[i] = elu_fp16(input[i]);
        }
    }
}

// ============================================================================
// Host Interface
// ============================================================================

extern "C" {

void cuda_kernel(const void* input, void* output, int N, int op_type) {
    const int threads = 256;
    int blocks;
    
    switch (op_type) {
        case 0: // FP32 Scalar
            blocks = (N + threads - 1) / threads;
            elu_fp32_scalar<<<blocks, threads>>>(
                reinterpret_cast<const float*>(input),
                reinterpret_cast<float*>(output),
                N
            );
            break;
            
        case 1: // FP32 Vec4
            blocks = (N / 4 + threads - 1) / threads;
            elu_fp32_vec4<<<blocks, threads>>>(
                reinterpret_cast<const float*>(input),
                reinterpret_cast<float*>(output),
                N
            );
            break;
            
        case 2: // FP16 Scalar
            blocks = (N + threads - 1) / threads;
            elu_fp16_scalar<<<blocks, threads>>>(
                reinterpret_cast<const half*>(input),
                reinterpret_cast<half*>(output),
                N
            );
            break;
            
        case 3: // FP16 Vec2
            blocks = (N / 2 + threads - 1) / threads;
            elu_fp16_vec2<<<blocks, threads>>>(
                reinterpret_cast<const half*>(input),
                reinterpret_cast<half*>(output),
                N
            );
            break;
            
        case 4: // FP16 Vec8 Unpack
            blocks = (N / 8 + threads - 1) / threads;
            elu_fp16_vec8_unpack<<<blocks, threads>>>(
                reinterpret_cast<const half*>(input),
                reinterpret_cast<half*>(output),
                N
            );
            break;
            
        case 5: // FP16 Vec8 Pack
            blocks = (N / 8 + threads - 1) / threads;
            elu_fp16_vec8_pack<<<blocks, threads>>>(
                reinterpret_cast<const half*>(input),
                reinterpret_cast<half*>(output),
                N
            );
            break;
            
        default:
            // Fallback to scalar
            blocks = (N + threads - 1) / threads;
            elu_fp32_scalar<<<blocks, threads>>>(
                reinterpret_cast<const float*>(input),
                reinterpret_cast<float*>(output),
                N
            );
            break;
    }
}

} // extern "C"
