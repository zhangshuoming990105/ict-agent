#include <cuda_runtime.h>
#include <cuda_fp16.h>

// ==================== FP32 Kernels ====================

// OP_TYPE 0: FP32 Scalar
__global__ void relu_f32_scalar(const float* __restrict__ x, 
                                float* __restrict__ out, 
                                int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        out[idx] = fmaxf(x[idx], 0.0f);
    }
}

// OP_TYPE 1: FP32 Vectorized (float4)
__global__ void relu_f32_vec4(const float* __restrict__ x, 
                               float* __restrict__ out, 
                               int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int vec_idx = idx * 4;
    
    if (vec_idx + 3 < N) {
        float4 val = reinterpret_cast<const float4*>(x)[idx];
        val.x = fmaxf(val.x, 0.0f);
        val.y = fmaxf(val.y, 0.0f);
        val.z = fmaxf(val.z, 0.0f);
        val.w = fmaxf(val.w, 0.0f);
        reinterpret_cast<float4*>(out)[idx] = val;
    } else if (vec_idx < N) {
        // Handle tail elements
        for (int i = vec_idx; i < N; i++) {
            out[i] = fmaxf(x[i], 0.0f);
        }
    }
}

// ==================== FP16 Kernels ====================

// OP_TYPE 2: FP16 Scalar
__global__ void relu_f16_scalar(const __half* __restrict__ x, 
                                __half* __restrict__ out, 
                                int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        __half val = x[idx];
        __half zero = __float2half(0.0f);
        out[idx] = __hgt(val, zero) ? val : zero;
    }
}

// OP_TYPE 3: FP16 Vectorized (half2)
__global__ void relu_f16_vec2(const __half* __restrict__ x, 
                               __half* __restrict__ out, 
                               int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int vec_idx = idx * 2;
    
    if (vec_idx + 1 < N) {
        __half2 val = reinterpret_cast<const __half2*>(x)[idx];
        __half2 zero = __float2half2_rn(0.0f);
        out[vec_idx] = __hgt(__low2half(val), __low2half(zero)) ? __low2half(val) : __low2half(zero);
        out[vec_idx + 1] = __hgt(__high2half(val), __high2half(zero)) ? __high2half(val) : __high2half(zero);
    } else if (vec_idx < N) {
        __half val = x[vec_idx];
        __half zero = __float2half(0.0f);
        out[vec_idx] = __hgt(val, zero) ? val : zero;
    }
}

// OP_TYPE 4: FP16 Vectorized x8 with unpack
__global__ void relu_f16_vec8_unpack(const __half* __restrict__ x, 
                                     __half* __restrict__ out, 
                                     int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int vec_idx = idx * 8;
    
    if (vec_idx + 7 < N) {
        // Load 8 half values using 4 float loads (128 bits total)
        float4 data = reinterpret_cast<const float4*>(x)[idx];
        
        // Cast to half2 for processing
        __half2* h2_ptr = reinterpret_cast<__half2*>(&data);
        __half2 zero = __float2half2_rn(0.0f);
        
        // Process 4 pairs
        __half2 r0 = __hmul2(__hgt2(h2_ptr[0], zero), h2_ptr[0]);
        __half2 r1 = __hmul2(__hgt2(h2_ptr[1], zero), h2_ptr[1]);
        __half2 r2 = __hmul2(__hgt2(h2_ptr[2], zero), h2_ptr[2]);
        __half2 r3 = __hmul2(__hgt2(h2_ptr[3], zero), h2_ptr[3]);
        
        // Store back
        float4 result;
        __half2* res_h2 = reinterpret_cast<__half2*>(&result);
        res_h2[0] = r0;
        res_h2[1] = r1;
        res_h2[2] = r2;
        res_h2[3] = r3;
        
        reinterpret_cast<float4*>(out)[idx] = result;
    } else if (vec_idx < N) {
        // Handle tail
        for (int i = vec_idx; i < N; i++) {
            __half val = x[i];
            __half zero = __float2half(0.0f);
            out[i] = __hgt(val, zero) ? val : zero;
        }
    }
}

// OP_TYPE 5: FP16 Vectorized x8 with packed operations
__global__ void relu_f16_vec8_pack(const __half* __restrict__ x, 
                                   __half* __restrict__ out, 
                                   int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int vec_idx = idx * 8;
    
    if (vec_idx + 7 < N) {
        // Load 128 bits (8 halfs)
        float4 data = reinterpret_cast<const float4*>(x)[idx];
        
        // Process as half2 vectors
        __half2* h2_data = reinterpret_cast<__half2*>(&data);
        __half2 zero = __float2half2_rn(0.0f);
        
        // Use __hmax2 for efficient packed max
        h2_data[0] = __hmax2(h2_data[0], zero);
        h2_data[1] = __hmax2(h2_data[1], zero);
        h2_data[2] = __hmax2(h2_data[2], zero);
        h2_data[3] = __hmax2(h2_data[3], zero);
        
        // Store result
        reinterpret_cast<float4*>(out)[idx] = data;
    } else if (vec_idx < N) {
        // Handle tail
        for (int i = vec_idx; i < N; i++) {
            __half val = x[i];
            __half zero = __float2half(0.0f);
            out[i] = __hgt(val, zero) ? val : zero;
        }
    }
}

// ==================== Host Launcher ====================

extern "C" void cuda_kernel(void* x_ptr, void* out_ptr, int N, int op_type) {
    const int threads = 256;
    int blocks;
    
    switch (op_type) {
        case 0: // FP32 scalar
            blocks = (N + threads - 1) / threads;
            relu_f32_scalar<<<blocks, threads>>>(
                reinterpret_cast<const float*>(x_ptr),
                reinterpret_cast<float*>(out_ptr),
                N
            );
            break;
            
        case 1: // FP32 vec4
            blocks = (N / 4 + threads - 1) / threads;
            relu_f32_vec4<<<blocks, threads>>>(
                reinterpret_cast<const float*>(x_ptr),
                reinterpret_cast<float*>(out_ptr),
                N
            );
            break;
            
        case 2: // FP16 scalar
            blocks = (N + threads - 1) / threads;
            relu_f16_scalar<<<blocks, threads>>>(
                reinterpret_cast<const __half*>(x_ptr),
                reinterpret_cast<__half*>(out_ptr),
                N
            );
            break;
            
        case 3: // FP16 vec2
            blocks = (N / 2 + threads - 1) / threads;
            relu_f16_vec2<<<blocks, threads>>>(
                reinterpret_cast<const __half*>(x_ptr),
                reinterpret_cast<__half*>(out_ptr),
                N
            );
            break;
            
        case 4: // FP16 vec8 unpack
            blocks = (N / 8 + threads - 1) / threads;
            relu_f16_vec8_unpack<<<blocks, threads>>>(
                reinterpret_cast<const __half*>(x_ptr),
                reinterpret_cast<__half*>(out_ptr),
                N
            );
            break;
            
        case 5: // FP16 vec8 pack
            blocks = (N / 8 + threads - 1) / threads;
            relu_f16_vec8_pack<<<blocks, threads>>>(
                reinterpret_cast<const __half*>(x_ptr),
                reinterpret_cast<__half*>(out_ptr),
                N
            );
            break;
            
        default:
            // Fallback to scalar
            blocks = (N + threads - 1) / threads;
            relu_f32_scalar<<<blocks, threads>>>(
                reinterpret_cast<const float*>(x_ptr),
                reinterpret_cast<float*>(out_ptr),
                N
            );
            break;
    }
}
