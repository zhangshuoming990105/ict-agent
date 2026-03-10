#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>
#include <cstdio>

// ============================================================================
// Utility functions for FP16 operations
// ============================================================================

__device__ __forceinline__ float half_to_float(__half h) {
    return __half2float(h);
}

__device__ __forceinline__ __half float_to_half(float f) {
    return __float2half(f);
}

__device__ __forceinline__ float2 half2_to_float2(__half2 h2) {
    float2 f2;
    f2.x = __half2float(__low2half(h2));
    f2.y = __half2float(__high2half(h2));
    return f2;
}

__device__ __forceinline__ __half2 float2_to_half2(float2 f2) {
    return __floats2half2_rn(f2.x, f2.y);
}

// ============================================================================
// Warp-level reduction utilities
// ============================================================================

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// ============================================================================
// Block-level reduction
// ============================================================================

__device__ __forceinline__ float block_reduce_sum(float val) {
    __shared__ float warp_sums[32];
    
    int lane = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;
    
    // Warp-level reduction
    val = warp_reduce_sum(val);
    
    // Write warp result
    if (lane == 0) {
        warp_sums[warp_id] = val;
    }
    __syncthreads();
    
    // Final reduction by first warp
    if (warp_id == 0) {
        val = (lane < (blockDim.x + 31) / 32) ? warp_sums[lane] : 0.0f;
        val = warp_reduce_sum(val);
    }
    
    return val;
}

// ============================================================================
// FP32 Kernels (Variant 0 and 1)
// ============================================================================

// Variant 0: Naive FP32
__global__ void layernorm_fp32_naive(
    const float* __restrict__ x,
    float* __restrict__ y,
    float gamma,
    float beta,
    int N,
    int K
) {
    int row = blockIdx.x;
    if (row >= N) return;

    const float* x_row = x + row * K;
    float* y_row = y + row * K;

    // Compute mean
    float sum = 0.0f;
    for (int i = threadIdx.x; i < K; i += blockDim.x) {
        sum += x_row[i];
    }
    
    sum = block_reduce_sum(sum);
    
    __shared__ float s_mean;
    if (threadIdx.x == 0) {
        s_mean = sum / K;
    }
    __syncthreads();
    
    float mean = s_mean;

    // Compute variance
    float var_sum = 0.0f;
    for (int i = threadIdx.x; i < K; i += blockDim.x) {
        float diff = x_row[i] - mean;
        var_sum += diff * diff;
    }
    
    var_sum = block_reduce_sum(var_sum);
    
    __shared__ float s_inv_std;
    if (threadIdx.x == 0) {
        float variance = var_sum / K;
        s_inv_std = rsqrtf(variance + 1e-5f);
    }
    __syncthreads();
    
    float inv_std = s_inv_std;

    // Normalize
    for (int i = threadIdx.x; i < K; i += blockDim.x) {
        float normalized = (x_row[i] - mean) * inv_std;
        y_row[i] = gamma * normalized + beta;
    }
}

// Variant 1: Vectorized FP32
__global__ void layernorm_fp32_vectorized(
    const float* __restrict__ x,
    float* __restrict__ y,
    float gamma,
    float beta,
    int N,
    int K
) {
    int row = blockIdx.x;
    if (row >= N) return;

    const float* x_row = x + row * K;
    float* y_row = y + row * K;

    // Compute mean with vectorized loads
    float sum = 0.0f;
    int vec_size = K / 4;
    const float4* x_vec = reinterpret_cast<const float4*>(x_row);
    
    for (int i = threadIdx.x; i < vec_size; i += blockDim.x) {
        float4 val = x_vec[i];
        sum += val.x + val.y + val.z + val.w;
    }
    
    // Handle remaining elements
    for (int i = vec_size * 4 + threadIdx.x; i < K; i += blockDim.x) {
        sum += x_row[i];
    }
    
    sum = block_reduce_sum(sum);
    
    __shared__ float s_mean;
    if (threadIdx.x == 0) {
        s_mean = sum / K;
    }
    __syncthreads();
    
    float mean = s_mean;

    // Compute variance with vectorized loads
    float var_sum = 0.0f;
    for (int i = threadIdx.x; i < vec_size; i += blockDim.x) {
        float4 val = x_vec[i];
        float diff_x = val.x - mean;
        float diff_y = val.y - mean;
        float diff_z = val.z - mean;
        float diff_w = val.w - mean;
        var_sum += diff_x * diff_x + diff_y * diff_y + diff_z * diff_z + diff_w * diff_w;
    }
    
    for (int i = vec_size * 4 + threadIdx.x; i < K; i += blockDim.x) {
        float diff = x_row[i] - mean;
        var_sum += diff * diff;
    }
    
    var_sum = block_reduce_sum(var_sum);
    
    __shared__ float s_inv_std;
    if (threadIdx.x == 0) {
        float variance = var_sum / K;
        s_inv_std = rsqrtf(variance + 1e-5f);
    }
    __syncthreads();
    
    float inv_std = s_inv_std;

    // Normalize with vectorized stores
    float4* y_vec = reinterpret_cast<float4*>(y_row);
    for (int i = threadIdx.x; i < vec_size; i += blockDim.x) {
        float4 val = x_vec[i];
        float4 result;
        result.x = gamma * (val.x - mean) * inv_std + beta;
        result.y = gamma * (val.y - mean) * inv_std + beta;
        result.z = gamma * (val.z - mean) * inv_std + beta;
        result.w = gamma * (val.w - mean) * inv_std + beta;
        y_vec[i] = result;
    }
    
    for (int i = vec_size * 4 + threadIdx.x; i < K; i += blockDim.x) {
        float normalized = (x_row[i] - mean) * inv_std;
        y_row[i] = gamma * normalized + beta;
    }
}

// ============================================================================
// FP16 Kernels (Variants 2-7)
// ============================================================================

// Variant 2: Naive FP16
__global__ void layernorm_fp16_naive(
    const __half* __restrict__ x,
    __half* __restrict__ y,
    float gamma,
    float beta,
    int N,
    int K
) {
    int row = blockIdx.x;
    if (row >= N) return;

    const __half* x_row = x + row * K;
    __half* y_row = y + row * K;

    // Compute mean (use FP32 accumulation)
    float sum = 0.0f;
    for (int i = threadIdx.x; i < K; i += blockDim.x) {
        sum += half_to_float(x_row[i]);
    }
    
    sum = block_reduce_sum(sum);
    
    __shared__ float s_mean;
    if (threadIdx.x == 0) {
        s_mean = sum / K;
    }
    __syncthreads();
    
    float mean = s_mean;

    // Compute variance
    float var_sum = 0.0f;
    for (int i = threadIdx.x; i < K; i += blockDim.x) {
        float val = half_to_float(x_row[i]);
        float diff = val - mean;
        var_sum += diff * diff;
    }
    
    var_sum = block_reduce_sum(var_sum);
    
    __shared__ float s_inv_std;
    if (threadIdx.x == 0) {
        float variance = var_sum / K;
        s_inv_std = rsqrtf(variance + 1e-5f);
    }
    __syncthreads();
    
    float inv_std = s_inv_std;

    // Normalize
    for (int i = threadIdx.x; i < K; i += blockDim.x) {
        float val = half_to_float(x_row[i]);
        float normalized = (val - mean) * inv_std;
        y_row[i] = float_to_half(gamma * normalized + beta);
    }
}

// Variant 4: FP16 Vec2
__global__ void layernorm_fp16_vec2(
    const __half* __restrict__ x,
    __half* __restrict__ y,
    float gamma,
    float beta,
    int N,
    int K
) {
    int row = blockIdx.x;
    if (row >= N) return;

    const __half* x_row = x + row * K;
    __half* y_row = y + row * K;

    // Compute mean with half2 vectorization
    float sum = 0.0f;
    int vec_size = K / 2;
    const __half2* x_vec = reinterpret_cast<const __half2*>(x_row);
    
    for (int i = threadIdx.x; i < vec_size; i += blockDim.x) {
        __half2 val = x_vec[i];
        float2 fval = half2_to_float2(val);
        sum += fval.x + fval.y;
    }
    
    if (K % 2 != 0 && threadIdx.x == 0) {
        sum += half_to_float(x_row[K - 1]);
    }
    
    sum = block_reduce_sum(sum);
    
    __shared__ float s_mean;
    if (threadIdx.x == 0) {
        s_mean = sum / K;
    }
    __syncthreads();
    
    float mean = s_mean;

    // Compute variance
    float var_sum = 0.0f;
    for (int i = threadIdx.x; i < vec_size; i += blockDim.x) {
        __half2 val = x_vec[i];
        float2 fval = half2_to_float2(val);
        float diff_x = fval.x - mean;
        float diff_y = fval.y - mean;
        var_sum += diff_x * diff_x + diff_y * diff_y;
    }
    
    if (K % 2 != 0 && threadIdx.x == 0) {
        float diff = half_to_float(x_row[K - 1]) - mean;
        var_sum += diff * diff;
    }
    
    var_sum = block_reduce_sum(var_sum);
    
    __shared__ float s_inv_std;
    if (threadIdx.x == 0) {
        float variance = var_sum / K;
        s_inv_std = rsqrtf(variance + 1e-5f);
    }
    __syncthreads();
    
    float inv_std = s_inv_std;

    // Normalize
    __half2* y_vec = reinterpret_cast<__half2*>(y_row);
    for (int i = threadIdx.x; i < vec_size; i += blockDim.x) {
        __half2 val = x_vec[i];
        float2 fval = half2_to_float2(val);
        float2 result;
        result.x = gamma * (fval.x - mean) * inv_std + beta;
        result.y = gamma * (fval.y - mean) * inv_std + beta;
        y_vec[i] = float2_to_half2(result);
    }
    
    if (K % 2 != 0 && threadIdx.x == 0) {
        float val = half_to_float(x_row[K - 1]);
        float normalized = (val - mean) * inv_std;
        y_row[K - 1] = float_to_half(gamma * normalized + beta);
    }
}

// Variant 5: FP16 Vec8
__global__ void layernorm_fp16_vec8(
    const __half* __restrict__ x,
    __half* __restrict__ y,
    float gamma,
    float beta,
    int N,
    int K
) {
    int row = blockIdx.x;
    if (row >= N) return;

    const __half* x_row = x + row * K;
    __half* y_row = y + row * K;

    // Compute mean with float4 (8 halfs)
    float sum = 0.0f;
    int vec_size = K / 8;
    const float4* x_vec = reinterpret_cast<const float4*>(x_row);
    
    for (int i = threadIdx.x; i < vec_size; i += blockDim.x) {
        float4 val = x_vec[i];
        __half2* h2_ptr = reinterpret_cast<__half2*>(&val);
        for (int j = 0; j < 4; j++) {
            float2 fval = half2_to_float2(h2_ptr[j]);
            sum += fval.x + fval.y;
        }
    }
    
    for (int i = vec_size * 8 + threadIdx.x; i < K; i += blockDim.x) {
        sum += half_to_float(x_row[i]);
    }
    
    sum = block_reduce_sum(sum);
    
    __shared__ float s_mean;
    if (threadIdx.x == 0) {
        s_mean = sum / K;
    }
    __syncthreads();
    
    float mean = s_mean;

    // Compute variance
    float var_sum = 0.0f;
    for (int i = threadIdx.x; i < vec_size; i += blockDim.x) {
        float4 val = x_vec[i];
        __half2* h2_ptr = reinterpret_cast<__half2*>(&val);
        for (int j = 0; j < 4; j++) {
            float2 fval = half2_to_float2(h2_ptr[j]);
            float diff_x = fval.x - mean;
            float diff_y = fval.y - mean;
            var_sum += diff_x * diff_x + diff_y * diff_y;
        }
    }
    
    for (int i = vec_size * 8 + threadIdx.x; i < K; i += blockDim.x) {
        float diff = half_to_float(x_row[i]) - mean;
        var_sum += diff * diff;
    }
    
    var_sum = block_reduce_sum(var_sum);
    
    __shared__ float s_inv_std;
    if (threadIdx.x == 0) {
        float variance = var_sum / K;
        s_inv_std = rsqrtf(variance + 1e-5f);
    }
    __syncthreads();
    
    float inv_std = s_inv_std;

    // Normalize
    float4* y_vec = reinterpret_cast<float4*>(y_row);
    for (int i = threadIdx.x; i < vec_size; i += blockDim.x) {
        float4 val = x_vec[i];
        float4 result;
        __half2* h2_in = reinterpret_cast<__half2*>(&val);
        __half2* h2_out = reinterpret_cast<__half2*>(&result);
        
        for (int j = 0; j < 4; j++) {
            float2 fval = half2_to_float2(h2_in[j]);
            float2 fresult;
            fresult.x = gamma * (fval.x - mean) * inv_std + beta;
            fresult.y = gamma * (fval.y - mean) * inv_std + beta;
            h2_out[j] = float2_to_half2(fresult);
        }
        y_vec[i] = result;
    }
    
    for (int i = vec_size * 8 + threadIdx.x; i < K; i += blockDim.x) {
        float val = half_to_float(x_row[i]);
        float normalized = (val - mean) * inv_std;
        y_row[i] = float_to_half(gamma * normalized + beta);
    }
}

// ============================================================================
// Main Kernel Dispatcher
// ============================================================================

extern "C" void cuda_kernel(
    void* x_ptr,
    void* y_ptr,
    float gamma,
    float beta,
    int N,
    int K,
    int variant_code
) {
    dim3 grid(N);
    dim3 block(256);

    // Dispatch based on variant_code
    if (variant_code == 0) {
        // FP32 Naive
        layernorm_fp32_naive<<<grid, block>>>(
            static_cast<float*>(x_ptr),
            static_cast<float*>(y_ptr),
            gamma, beta, N, K
        );
    } else if (variant_code == 1) {
        // FP32 Vectorized
        layernorm_fp32_vectorized<<<grid, block>>>(
            static_cast<float*>(x_ptr),
            static_cast<float*>(y_ptr),
            gamma, beta, N, K
        );
    } else if (variant_code == 2) {
        // FP16 Naive
        layernorm_fp16_naive<<<grid, block>>>(
            static_cast<__half*>(x_ptr),
            static_cast<__half*>(y_ptr),
            gamma, beta, N, K
        );
    } else if (variant_code == 3) {
        // FP16 Naive with FP32 Accum (same as variant 2)
        layernorm_fp16_naive<<<grid, block>>>(
            static_cast<__half*>(x_ptr),
            static_cast<__half*>(y_ptr),
            gamma, beta, N, K
        );
    } else if (variant_code == 4) {
        // FP16 Vec2
        layernorm_fp16_vec2<<<grid, block>>>(
            static_cast<__half*>(x_ptr),
            static_cast<__half*>(y_ptr),
            gamma, beta, N, K
        );
    } else if (variant_code == 5) {
        // FP16 Vec8
        layernorm_fp16_vec8<<<grid, block>>>(
            static_cast<__half*>(x_ptr),
            static_cast<__half*>(y_ptr),
            gamma, beta, N, K
        );
    } else if (variant_code == 6) {
        // FP16 Pack FP16 (use vec8)
        layernorm_fp16_vec8<<<grid, block>>>(
            static_cast<__half*>(x_ptr),
            static_cast<__half*>(y_ptr),
            gamma, beta, N, K
        );
    } else if (variant_code == 7) {
        // FP16 Pack FP32 (use vec8)
        layernorm_fp16_vec8<<<grid, block>>>(
            static_cast<__half*>(x_ptr),
            static_cast<__half*>(y_ptr),
            gamma, beta, N, K
        );
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA kernel launch error: %s\n", cudaGetErrorString(err));
    }
}
