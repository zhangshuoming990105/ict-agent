#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>

#define WARP_SIZE 32
#define EPS 1e-6f

// ===========================
// FP32 Variants (0-1)
// ===========================

// Variant 0: FP32 Naive
__global__ void rmsnorm_f32_naive(const float* __restrict__ x, 
                                   float* __restrict__ y, 
                                   int N, int K) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= N) return;
    
    const float* x_row = x + row * K;
    float* y_row = y + row * K;
    
    // Compute sum of squares
    float sum_sq = 0.0f;
    for (int i = 0; i < K; i++) {
        float val = x_row[i];
        sum_sq += val * val;
    }
    
    float rms = rsqrtf(sum_sq / K + EPS);
    
    // Normalize
    for (int i = 0; i < K; i++) {
        y_row[i] = x_row[i] * rms;
    }
}

// Variant 1: FP32 Vectorized (float4)
__global__ void rmsnorm_f32_vec(const float* __restrict__ x, 
                                 float* __restrict__ y, 
                                 int N, int K) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= N) return;
    
    const float4* x_row = reinterpret_cast<const float4*>(x + row * K);
    float4* y_row = reinterpret_cast<float4*>(y + row * K);
    int K4 = K / 4;
    
    // Compute sum of squares with vectorized loads
    float sum_sq = 0.0f;
    for (int i = 0; i < K4; i++) {
        float4 val = x_row[i];
        sum_sq += val.x * val.x + val.y * val.y + val.z * val.z + val.w * val.w;
    }
    
    float rms = rsqrtf(sum_sq / K + EPS);
    
    // Normalize with vectorized stores
    for (int i = 0; i < K4; i++) {
        float4 val = x_row[i];
        float4 out;
        out.x = val.x * rms;
        out.y = val.y * rms;
        out.z = val.z * rms;
        out.w = val.w * rms;
        y_row[i] = out;
    }
}

// ===========================
// FP16 Variants (2-8)
// ===========================

// Variant 2: FP16 Naive (FP16 accumulation)
__global__ void rmsnorm_f16_naive(const half* __restrict__ x, 
                                   half* __restrict__ y, 
                                   int N, int K) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= N) return;
    
    const half* x_row = x + row * K;
    half* y_row = y + row * K;
    
    float sum_sq = 0.0f;
    for (int i = 0; i < K; i++) {
        float val = __half2float(x_row[i]);
        sum_sq += val * val;
    }
    
    float rms = rsqrtf(sum_sq / K + EPS);
    
    for (int i = 0; i < K; i++) {
        float val = __half2float(x_row[i]);
        y_row[i] = __float2half(val * rms);
    }
}

// Variant 3: FP16 Naive with FP32 accumulation (same as variant 2 in practice)
__global__ void rmsnorm_f16_naive_f32(const half* __restrict__ x, 
                                       half* __restrict__ y, 
                                       int N, int K) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= N) return;
    
    const half* x_row = x + row * K;
    half* y_row = y + row * K;
    
    float sum_sq = 0.0f;
    for (int i = 0; i < K; i++) {
        float val = __half2float(x_row[i]);
        sum_sq += val * val;
    }
    
    float rms = rsqrtf(sum_sq / K + EPS);
    
    for (int i = 0; i < K; i++) {
        float val = __half2float(x_row[i]);
        y_row[i] = __float2half(val * rms);
    }
}

// Variant 4: FP16 Vec2 (half2)
__global__ void rmsnorm_f16_vec2(const half* __restrict__ x, 
                                  half* __restrict__ y, 
                                  int N, int K) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= N) return;
    
    const half2* x_row = reinterpret_cast<const half2*>(x + row * K);
    half2* y_row = reinterpret_cast<half2*>(y + row * K);
    int K2 = K / 2;
    
    float sum_sq = 0.0f;
    for (int i = 0; i < K2; i++) {
        half2 val = x_row[i];
        float2 val_f = __half22float2(val);
        sum_sq += val_f.x * val_f.x + val_f.y * val_f.y;
    }
    
    float rms = rsqrtf(sum_sq / K + EPS);
    half2 rms_h2 = __float2half2_rn(rms);
    
    for (int i = 0; i < K2; i++) {
        half2 val = x_row[i];
        y_row[i] = __hmul2(val, rms_h2);
    }
}

// Variant 5: FP16 Vec8 with FP16 accumulation
__global__ void rmsnorm_f16_vec8_f16(const half* __restrict__ x, 
                                      half* __restrict__ y, 
                                      int N, int K) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= N) return;
    
    const float4* x_row = reinterpret_cast<const float4*>(x + row * K);
    float4* y_row = reinterpret_cast<float4*>(y + row * K);
    int K8 = K / 8;
    
    float sum_sq = 0.0f;
    for (int i = 0; i < K8; i++) {
        float4 val_packed = x_row[i];
        half2* val_h2 = reinterpret_cast<half2*>(&val_packed);
        
        for (int j = 0; j < 4; j++) {
            float2 val_f = __half22float2(val_h2[j]);
            sum_sq += val_f.x * val_f.x + val_f.y * val_f.y;
        }
    }
    
    float rms = rsqrtf(sum_sq / K + EPS);
    half2 rms_h2 = __float2half2_rn(rms);
    
    for (int i = 0; i < K8; i++) {
        float4 val_packed = x_row[i];
        half2* val_h2 = reinterpret_cast<half2*>(&val_packed);
        float4 out_packed;
        half2* out_h2 = reinterpret_cast<half2*>(&out_packed);
        
        for (int j = 0; j < 4; j++) {
            out_h2[j] = __hmul2(val_h2[j], rms_h2);
        }
        y_row[i] = out_packed;
    }
}

// Variant 6: FP16 Vec8 with FP32 accumulation (more stable)
__global__ void rmsnorm_f16_vec8_f32(const half* __restrict__ x, 
                                      half* __restrict__ y, 
                                      int N, int K) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= N) return;
    
    const float4* x_row = reinterpret_cast<const float4*>(x + row * K);
    float4* y_row = reinterpret_cast<float4*>(y + row * K);
    int K8 = K / 8;
    
    float sum_sq = 0.0f;
    for (int i = 0; i < K8; i++) {
        float4 val_packed = x_row[i];
        half2* val_h2 = reinterpret_cast<half2*>(&val_packed);
        
        for (int j = 0; j < 4; j++) {
            float2 val_f = __half22float2(val_h2[j]);
            sum_sq += val_f.x * val_f.x + val_f.y * val_f.y;
        }
    }
    
    float rms = rsqrtf(sum_sq / K + EPS);
    half2 rms_h2 = __float2half2_rn(rms);
    
    for (int i = 0; i < K8; i++) {
        float4 val_packed = x_row[i];
        half2* val_h2 = reinterpret_cast<half2*>(&val_packed);
        float4 out_packed;
        half2* out_h2 = reinterpret_cast<half2*>(&out_packed);
        
        for (int j = 0; j < 4; j++) {
            out_h2[j] = __hmul2(val_h2[j], rms_h2);
        }
        y_row[i] = out_packed;
    }
}

// Variant 7: FP16 Pack with warp reduction (FP16)
__global__ void rmsnorm_f16_pack_f16(const half* __restrict__ x, 
                                      half* __restrict__ y, 
                                      int N, int K) {
    int row = blockIdx.x;
    if (row >= N) return;
    
    const float4* x_row = reinterpret_cast<const float4*>(x + row * K);
    float4* y_row = reinterpret_cast<float4*>(y + row * K);
    int K8 = K / 8;
    
    // Each thread computes partial sum
    float sum_sq = 0.0f;
    for (int i = threadIdx.x; i < K8; i += blockDim.x) {
        float4 val_packed = x_row[i];
        half2* val_h2 = reinterpret_cast<half2*>(&val_packed);
        
        for (int j = 0; j < 4; j++) {
            float2 val_f = __half22float2(val_h2[j]);
            sum_sq += val_f.x * val_f.x + val_f.y * val_f.y;
        }
    }
    
    // Warp reduction
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset);
    }
    
    // Broadcast RMS to all threads
    float rms;
    if (threadIdx.x == 0) {
        rms = rsqrtf(sum_sq / K + EPS);
    }
    rms = __shfl_sync(0xffffffff, rms, 0);
    half2 rms_h2 = __float2half2_rn(rms);
    
    // Normalize
    for (int i = threadIdx.x; i < K8; i += blockDim.x) {
        float4 val_packed = x_row[i];
        half2* val_h2 = reinterpret_cast<half2*>(&val_packed);
        float4 out_packed;
        half2* out_h2 = reinterpret_cast<half2*>(&out_packed);
        
        for (int j = 0; j < 4; j++) {
            out_h2[j] = __hmul2(val_h2[j], rms_h2);
        }
        y_row[i] = out_packed;
    }
}

// Variant 8: FP16 Pack with block reduction (FP32 accumulation)
__global__ void rmsnorm_f16_pack_f32(const half* __restrict__ x, 
                                      half* __restrict__ y, 
                                      int N, int K) {
    int row = blockIdx.x;
    if (row >= N) return;
    
    const float4* x_row = reinterpret_cast<const float4*>(x + row * K);
    float4* y_row = reinterpret_cast<float4*>(y + row * K);
    int K8 = K / 8;
    
    __shared__ float shared_sum[32]; // For warp-level reduction results
    
    // Each thread computes partial sum
    float sum_sq = 0.0f;
    for (int i = threadIdx.x; i < K8; i += blockDim.x) {
        float4 val_packed = x_row[i];
        half2* val_h2 = reinterpret_cast<half2*>(&val_packed);
        
        for (int j = 0; j < 4; j++) {
            float2 val_f = __half22float2(val_h2[j]);
            sum_sq += val_f.x * val_f.x + val_f.y * val_f.y;
        }
    }
    
    // Warp reduction
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset);
    }
    
    // Write warp results to shared memory
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;
    if (lane_id == 0) {
        shared_sum[warp_id] = sum_sq;
    }
    __syncthreads();
    
    // Final reduction by first warp
    if (warp_id == 0) {
        int num_warps = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;
        sum_sq = (lane_id < num_warps) ? shared_sum[lane_id] : 0.0f;
        
        #pragma unroll
        for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
            sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset);
        }
    }
    
    // Broadcast RMS to all threads
    __shared__ float rms_shared;
    if (threadIdx.x == 0) {
        rms_shared = rsqrtf(sum_sq / K + EPS);
    }
    __syncthreads();
    float rms = rms_shared;
    half2 rms_h2 = __float2half2_rn(rms);
    
    // Normalize
    for (int i = threadIdx.x; i < K8; i += blockDim.x) {
        float4 val_packed = x_row[i];
        half2* val_h2 = reinterpret_cast<half2*>(&val_packed);
        float4 out_packed;
        half2* out_h2 = reinterpret_cast<half2*>(&out_packed);
        
        for (int j = 0; j < 4; j++) {
            out_h2[j] = __hmul2(val_h2[j], rms_h2);
        }
        y_row[i] = out_packed;
    }
}

// ===========================
// Dispatcher Kernel
// ===========================

extern "C" void cuda_kernel(void* x, void* y, float g, int N, int K, int variant_code) {
    // Dispatch based on variant code
    if (variant_code == 0) {
        // FP32 Naive
        int threads = 256;
        int blocks = (N + threads - 1) / threads;
        rmsnorm_f32_naive<<<blocks, threads>>>(
            reinterpret_cast<float*>(x),
            reinterpret_cast<float*>(y),
            N, K
        );
    } else if (variant_code == 1) {
        // FP32 Vectorized
        int threads = 256;
        int blocks = (N + threads - 1) / threads;
        rmsnorm_f32_vec<<<blocks, threads>>>(
            reinterpret_cast<float*>(x),
            reinterpret_cast<float*>(y),
            N, K
        );
    } else if (variant_code == 2) {
        // FP16 Naive
        int threads = 256;
        int blocks = (N + threads - 1) / threads;
        rmsnorm_f16_naive<<<blocks, threads>>>(
            reinterpret_cast<half*>(x),
            reinterpret_cast<half*>(y),
            N, K
        );
    } else if (variant_code == 3) {
        // FP16 Naive F32 Accum
        int threads = 256;
        int blocks = (N + threads - 1) / threads;
        rmsnorm_f16_naive_f32<<<blocks, threads>>>(
            reinterpret_cast<half*>(x),
            reinterpret_cast<half*>(y),
            N, K
        );
    } else if (variant_code == 4) {
        // FP16 Vec2
        int threads = 256;
        int blocks = (N + threads - 1) / threads;
        rmsnorm_f16_vec2<<<blocks, threads>>>(
            reinterpret_cast<half*>(x),
            reinterpret_cast<half*>(y),
            N, K
        );
    } else if (variant_code == 5) {
        // FP16 Vec8 F16
        int threads = 256;
        int blocks = (N + threads - 1) / threads;
        rmsnorm_f16_vec8_f16<<<blocks, threads>>>(
            reinterpret_cast<half*>(x),
            reinterpret_cast<half*>(y),
            N, K
        );
    } else if (variant_code == 6) {
        // FP16 Vec8 F32
        int threads = 256;
        int blocks = (N + threads - 1) / threads;
        rmsnorm_f16_vec8_f32<<<blocks, threads>>>(
            reinterpret_cast<half*>(x),
            reinterpret_cast<half*>(y),
            N, K
        );
    } else if (variant_code == 7) {
        // FP16 Pack F16
        int threads = 32; // Single warp per row
        int blocks = N;
        rmsnorm_f16_pack_f16<<<blocks, threads>>>(
            reinterpret_cast<half*>(x),
            reinterpret_cast<half*>(y),
            N, K
        );
    } else if (variant_code == 8) {
        // FP16 Pack F32
        int threads = 256; // Multiple warps per row
        int blocks = N;
        rmsnorm_f16_pack_f32<<<blocks, threads>>>(
            reinterpret_cast<half*>(x),
            reinterpret_cast<half*>(y),
            N, K
        );
    }
}
