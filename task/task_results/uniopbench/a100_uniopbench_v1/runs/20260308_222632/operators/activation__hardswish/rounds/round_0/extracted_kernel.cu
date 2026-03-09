#include <cuda_runtime.h>
#include <cuda_fp16.h>

// Hardswish formula: x * min(max(x + 3, 0), 6) / 6

// ============================================================================
// FP32 Scalar Implementation (OP_TYPE = 0)
// ============================================================================
__device__ __forceinline__ float hardswish_f32(float x) {
    float relu6_val = fminf(fmaxf(x + 3.0f, 0.0f), 6.0f);
    return x * relu6_val / 6.0f;
}

// ============================================================================
// FP32 Vectorized x4 Implementation (OP_TYPE = 1)
// ============================================================================
__device__ __forceinline__ float4 hardswish_f32x4(float4 x) {
    float4 result;
    result.x = hardswish_f32(x.x);
    result.y = hardswish_f32(x.y);
    result.z = hardswish_f32(x.z);
    result.w = hardswish_f32(x.w);
    return result;
}

// ============================================================================
// FP16 Scalar Implementation (OP_TYPE = 2)
// ============================================================================
__device__ __forceinline__ half hardswish_f16(half x) {
    half three = __float2half(3.0f);
    half zero = __float2half(0.0f);
    half six = __float2half(6.0f);
    half relu6_val = __hmin(__hmax(__hadd(x, three), zero), six);
    return __hdiv(__hmul(x, relu6_val), six);
}

// ============================================================================
// FP16 Vectorized x2 Implementation (OP_TYPE = 3)
// ============================================================================
__device__ __forceinline__ half2 hardswish_f16x2(half2 x) {
    half2 three = __float2half2_rn(3.0f);
    half2 zero = __float2half2_rn(0.0f);
    half2 six = __float2half2_rn(6.0f);
    half2 relu6_val = __hmin2(__hmax2(__hadd2(x, three), zero), six);
    return __h2div(__hmul2(x, relu6_val), six);
}

// ============================================================================
// FP16 Vectorized x8 Unpack Implementation (OP_TYPE = 4)
// ============================================================================
__device__ __forceinline__ void hardswish_f16x8_unpack(
    const half* input, half* output
) {
    half2 v0 = *reinterpret_cast<const half2*>(input + 0);
    half2 v1 = *reinterpret_cast<const half2*>(input + 2);
    half2 v2 = *reinterpret_cast<const half2*>(input + 4);
    half2 v3 = *reinterpret_cast<const half2*>(input + 6);
    
    v0 = hardswish_f16x2(v0);
    v1 = hardswish_f16x2(v1);
    v2 = hardswish_f16x2(v2);
    v3 = hardswish_f16x2(v3);
    
    *reinterpret_cast<half2*>(output + 0) = v0;
    *reinterpret_cast<half2*>(output + 2) = v1;
    *reinterpret_cast<half2*>(output + 4) = v2;
    *reinterpret_cast<half2*>(output + 6) = v3;
}

// ============================================================================
// FP16 Vectorized x8 Pack Implementation (OP_TYPE = 5)
// ============================================================================
__device__ __forceinline__ void hardswish_f16x8_pack(
    const half* input, half* output
) {
    // Load as uint4 for maximum memory bandwidth (128-bit load)
    uint4 data = *reinterpret_cast<const uint4*>(input);
    
    half2* h2_ptr = reinterpret_cast<half2*>(&data);
    h2_ptr[0] = hardswish_f16x2(h2_ptr[0]);
    h2_ptr[1] = hardswish_f16x2(h2_ptr[1]);
    h2_ptr[2] = hardswish_f16x2(h2_ptr[2]);
    h2_ptr[3] = hardswish_f16x2(h2_ptr[3]);
    
    // Store as uint4 (128-bit store)
    *reinterpret_cast<uint4*>(output) = data;
}

// ============================================================================
// Unified Kernel
// ============================================================================
__global__ void hardswish_kernel(
    const void* input,
    void* output,
    int N,
    int op_type
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (op_type == 0) {
        // FP32 Scalar
        const float* in_f32 = reinterpret_cast<const float*>(input);
        float* out_f32 = reinterpret_cast<float*>(output);
        
        if (idx < N) {
            out_f32[idx] = hardswish_f32(in_f32[idx]);
        }
    }
    else if (op_type == 1) {
        // FP32 Vectorized x4
        const float4* in_f32x4 = reinterpret_cast<const float4*>(input);
        float4* out_f32x4 = reinterpret_cast<float4*>(output);
        int N_vec = N / 4;
        
        if (idx < N_vec) {
            out_f32x4[idx] = hardswish_f32x4(in_f32x4[idx]);
        }
    }
    else if (op_type == 2) {
        // FP16 Scalar
        const half* in_f16 = reinterpret_cast<const half*>(input);
        half* out_f16 = reinterpret_cast<half*>(output);
        
        if (idx < N) {
            out_f16[idx] = hardswish_f16(in_f16[idx]);
        }
    }
    else if (op_type == 3) {
        // FP16 Vectorized x2
        const half2* in_f16x2 = reinterpret_cast<const half2*>(input);
        half2* out_f16x2 = reinterpret_cast<half2*>(output);
        int N_vec = N / 2;
        
        if (idx < N_vec) {
            out_f16x2[idx] = hardswish_f16x2(in_f16x2[idx]);
        }
    }
    else if (op_type == 4) {
        // FP16 Vectorized x8 Unpack
        const half* in_f16 = reinterpret_cast<const half*>(input);
        half* out_f16 = reinterpret_cast<half*>(output);
        int idx8 = idx * 8;
        
        if (idx8 + 7 < N) {
            hardswish_f16x8_unpack(in_f16 + idx8, out_f16 + idx8);
        }
    }
    else if (op_type == 5) {
        // FP16 Vectorized x8 Pack
        const half* in_f16 = reinterpret_cast<const half*>(input);
        half* out_f16 = reinterpret_cast<half*>(output);
        int idx8 = idx * 8;
        
        if (idx8 + 7 < N) {
            hardswish_f16x8_pack(in_f16 + idx8, out_f16 + idx8);
        }
    }
}

// ============================================================================
// Host Launcher (Expected by framework)
// ============================================================================
extern "C" void cuda_kernel(
    const void* input,
    void* output,
    int N,
    int op_type
) {
    // Determine grid and block dimensions based on op_type
    int threads_per_block = 256;
    int elements_per_thread;
    
    if (op_type == 1) {
        // FP32x4: each thread processes 4 elements
        elements_per_thread = 4;
    } else if (op_type == 3) {
        // FP16x2: each thread processes 2 elements
        elements_per_thread = 2;
    } else if (op_type == 4 || op_type == 5) {
        // FP16x8: each thread processes 8 elements
        elements_per_thread = 8;
    } else {
        // Scalar: each thread processes 1 element
        elements_per_thread = 1;
    }
    
    int num_elements_to_process = (N + elements_per_thread - 1) / elements_per_thread;
    int num_blocks = (num_elements_to_process + threads_per_block - 1) / threads_per_block;
    
    hardswish_kernel<<<num_blocks, threads_per_block>>>(input, output, N, op_type);
}
