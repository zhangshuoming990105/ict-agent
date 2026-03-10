#include <cuda_runtime.h>
#include <cuda_fp16.h>

// FP32 基础版本
__global__ void elementwise_add_f32(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ output,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = A[idx] + B[idx];
    }
}

// FP32 向量化 x4
__global__ void elementwise_add_f32x4(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ output,
    int size
) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    
    if (idx + 3 < size) {
        // Vectorized load/store using float4
        float4 a = *reinterpret_cast<const float4*>(&A[idx]);
        float4 b = *reinterpret_cast<const float4*>(&B[idx]);
        
        float4 result;
        result.x = a.x + b.x;
        result.y = a.y + b.y;
        result.z = a.z + b.z;
        result.w = a.w + b.w;
        
        *reinterpret_cast<float4*>(&output[idx]) = result;
    } else {
        // Handle remaining elements
        for (int i = idx; i < size && i < idx + 4; i++) {
            output[i] = A[i] + B[i];
        }
    }
}

// FP16 基础版本
__global__ void elementwise_add_f16(
    const __half* __restrict__ A,
    const __half* __restrict__ B,
    __half* __restrict__ output,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = __hadd(A[idx], B[idx]);
    }
}

// FP16 向量化 x2
__global__ void elementwise_add_f16x2(
    const __half* __restrict__ A,
    const __half* __restrict__ B,
    __half* __restrict__ output,
    int size
) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
    
    if (idx + 1 < size) {
        __half2 a = *reinterpret_cast<const __half2*>(&A[idx]);
        __half2 b = *reinterpret_cast<const __half2*>(&B[idx]);
        __half2 result = __hadd2(a, b);
        *reinterpret_cast<__half2*>(&output[idx]) = result;
    } else if (idx < size) {
        output[idx] = __hadd(A[idx], B[idx]);
    }
}

// FP16 向量化 x8
__global__ void elementwise_add_f16x8(
    const __half* __restrict__ A,
    const __half* __restrict__ B,
    __half* __restrict__ output,
    int size
) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 8;
    
    if (idx + 7 < size) {
        // Load 8 half values (128 bits) at once
        float4 a = *reinterpret_cast<const float4*>(&A[idx]);
        float4 b = *reinterpret_cast<const float4*>(&B[idx]);
        
        // Process as 4 x half2
        __half2* a_h2 = reinterpret_cast<__half2*>(&a);
        __half2* b_h2 = reinterpret_cast<__half2*>(&b);
        
        float4 result;
        __half2* result_h2 = reinterpret_cast<__half2*>(&result);
        
        result_h2[0] = __hadd2(a_h2[0], b_h2[0]);
        result_h2[1] = __hadd2(a_h2[1], b_h2[1]);
        result_h2[2] = __hadd2(a_h2[2], b_h2[2]);
        result_h2[3] = __hadd2(a_h2[3], b_h2[3]);
        
        *reinterpret_cast<float4*>(&output[idx]) = result;
    } else {
        // Handle remaining elements
        for (int i = idx; i < size && i < idx + 8; i++) {
            output[i] = __hadd(A[i], B[i]);
        }
    }
}

// FP16 向量化 x8 pack (alternative implementation)
__global__ void elementwise_add_f16x8_pack(
    const __half* __restrict__ A,
    const __half* __restrict__ B,
    __half* __restrict__ output,
    int size
) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 8;
    
    if (idx + 7 < size) {
        // Use uint4 for 128-bit aligned loads (8 halfs = 128 bits)
        uint4 a_data = *reinterpret_cast<const uint4*>(&A[idx]);
        uint4 b_data = *reinterpret_cast<const uint4*>(&B[idx]);
        
        __half2* a_h2 = reinterpret_cast<__half2*>(&a_data);
        __half2* b_h2 = reinterpret_cast<__half2*>(&b_data);
        
        uint4 result_data;
        __half2* result_h2 = reinterpret_cast<__half2*>(&result_data);
        
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            result_h2[i] = __hadd2(a_h2[i], b_h2[i]);
        }
        
        *reinterpret_cast<uint4*>(&output[idx]) = result_data;
    } else {
        // Handle remaining elements
        for (int i = idx; i < size && i < idx + 8; i++) {
            output[i] = __hadd(A[i], B[i]);
        }
    }
}

// Unified launcher function - named cuda_kernel as expected by the framework
extern "C" void cuda_kernel(
    void* A,
    void* B,
    void* output,
    int size,
    int op_type
) {
    const int threads = 256;
    int blocks;
    
    switch(op_type) {
        case 0: // elementwise_add_f32
            blocks = (size + threads - 1) / threads;
            elementwise_add_f32<<<blocks, threads>>>(
                reinterpret_cast<const float*>(A),
                reinterpret_cast<const float*>(B),
                reinterpret_cast<float*>(output),
                size
            );
            break;
            
        case 1: // elementwise_add_f32x4
            blocks = (size / 4 + threads - 1) / threads;
            elementwise_add_f32x4<<<blocks, threads>>>(
                reinterpret_cast<const float*>(A),
                reinterpret_cast<const float*>(B),
                reinterpret_cast<float*>(output),
                size
            );
            break;
            
        case 2: // elementwise_add_f16
            blocks = (size + threads - 1) / threads;
            elementwise_add_f16<<<blocks, threads>>>(
                reinterpret_cast<const __half*>(A),
                reinterpret_cast<const __half*>(B),
                reinterpret_cast<__half*>(output),
                size
            );
            break;
            
        case 3: // elementwise_add_f16x2
            blocks = (size / 2 + threads - 1) / threads;
            elementwise_add_f16x2<<<blocks, threads>>>(
                reinterpret_cast<const __half*>(A),
                reinterpret_cast<const __half*>(B),
                reinterpret_cast<__half*>(output),
                size
            );
            break;
            
        case 4: // elementwise_add_f16x8
            blocks = (size / 8 + threads - 1) / threads;
            elementwise_add_f16x8<<<blocks, threads>>>(
                reinterpret_cast<const __half*>(A),
                reinterpret_cast<const __half*>(B),
                reinterpret_cast<__half*>(output),
                size
            );
            break;
            
        case 5: // elementwise_add_f16x8_pack
            blocks = (size / 8 + threads - 1) / threads;
            elementwise_add_f16x8_pack<<<blocks, threads>>>(
                reinterpret_cast<const __half*>(A),
                reinterpret_cast<const __half*>(B),
                reinterpret_cast<__half*>(output),
                size
            );
            break;
            
        default:
            // Fallback to basic f32
            blocks = (size + threads - 1) / threads;
            elementwise_add_f32<<<blocks, threads>>>(
                reinterpret_cast<const float*>(A),
                reinterpret_cast<const float*>(B),
                reinterpret_cast<float*>(output),
                size
            );
            break;
    }
}
