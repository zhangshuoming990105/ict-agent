#include <cuda_runtime.h>
#include <cuda_fp16.h>

// ============================================================================
// GEMV Kernel: y = A * x
// A: (M, K) matrix
// x: (K, 1) vector
// y: (M, 1) output vector
// ============================================================================

// ----------------------------------------------------------------------------
// FP32 Kernels
// ----------------------------------------------------------------------------

// FP32 K=16 specialized kernel
__global__ void gemv_k16_fp32(const float* __restrict__ A,
                              const float* __restrict__ x,
                              float* __restrict__ y,
                              int M, int K) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M) {
        float sum = 0.0f;
        const float* a_row = A + row * K;
        
        #pragma unroll
        for (int k = 0; k < 16; k++) {
            sum += a_row[k] * x[k];
        }
        
        y[row] = sum;
    }
}

// FP32 general kernel for K=32, 64, or any multiple of 32
__global__ void gemv_k32_fp32(const float* __restrict__ A,
                              const float* __restrict__ x,
                              float* __restrict__ y,
                              int M, int K) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M) {
        float sum = 0.0f;
        const float* a_row = A + row * K;
        
        // Process 4 elements at a time for better memory coalescing
        int k = 0;
        for (; k + 3 < K; k += 4) {
            sum += a_row[k] * x[k];
            sum += a_row[k + 1] * x[k + 1];
            sum += a_row[k + 2] * x[k + 2];
            sum += a_row[k + 3] * x[k + 3];
        }
        
        // Handle remaining elements
        for (; k < K; k++) {
            sum += a_row[k] * x[k];
        }
        
        y[row] = sum;
    }
}

// FP32 vectorized kernel for K=128+ (using float4 loads)
__global__ void gemv_k128_fp32(const float* __restrict__ A,
                               const float* __restrict__ x,
                               float* __restrict__ y,
                               int M, int K) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M) {
        float sum = 0.0f;
        const float* a_row = A + row * K;
        
        // Use float4 for vectorized loads (128-bit loads)
        int k = 0;
        int K_vec = (K / 4) * 4;
        
        for (; k < K_vec; k += 4) {
            float4 a_vec = *reinterpret_cast<const float4*>(&a_row[k]);
            float4 x_vec = *reinterpret_cast<const float4*>(&x[k]);
            
            sum += a_vec.x * x_vec.x;
            sum += a_vec.y * x_vec.y;
            sum += a_vec.z * x_vec.z;
            sum += a_vec.w * x_vec.w;
        }
        
        // Handle remaining elements
        for (; k < K; k++) {
            sum += a_row[k] * x[k];
        }
        
        y[row] = sum;
    }
}

// ----------------------------------------------------------------------------
// FP16 Kernels
// ----------------------------------------------------------------------------

// FP16 K=16 specialized kernel
__global__ void gemv_k16_fp16(const half* __restrict__ A,
                              const half* __restrict__ x,
                              half* __restrict__ y,
                              int M, int K) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M) {
        float sum = 0.0f;
        const half* a_row = A + row * K;
        
        #pragma unroll
        for (int k = 0; k < 16; k++) {
            sum += __half2float(a_row[k]) * __half2float(x[k]);
        }
        
        y[row] = __float2half(sum);
    }
}

// FP16 general kernel for K=32, 64, or any multiple of 32
__global__ void gemv_k32_fp16(const half* __restrict__ A,
                              const half* __restrict__ x,
                              half* __restrict__ y,
                              int M, int K) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M) {
        float sum = 0.0f;
        const half* a_row = A + row * K;
        
        // Process with half2 for better performance
        int k = 0;
        int K_half2 = (K / 2) * 2;
        
        for (; k < K_half2; k += 2) {
            half2 a_vec = *reinterpret_cast<const half2*>(&a_row[k]);
            half2 x_vec = *reinterpret_cast<const half2*>(&x[k]);
            
            sum += __half2float(a_vec.x) * __half2float(x_vec.x);
            sum += __half2float(a_vec.y) * __half2float(x_vec.y);
        }
        
        // Handle remaining element
        if (k < K) {
            sum += __half2float(a_row[k]) * __half2float(x[k]);
        }
        
        y[row] = __float2half(sum);
    }
}

// FP16 vectorized kernel for K=128+ (using uint4 for half8 loads)
__global__ void gemv_k128_fp16(const half* __restrict__ A,
                               const half* __restrict__ x,
                               half* __restrict__ y,
                               int M, int K) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M) {
        float sum = 0.0f;
        const half* a_row = A + row * K;
        
        // Use uint4 for 128-bit loads (8 half values)
        int k = 0;
        int K_vec = (K / 8) * 8;
        
        for (; k < K_vec; k += 8) {
            uint4 a_vec = *reinterpret_cast<const uint4*>(&a_row[k]);
            uint4 x_vec = *reinterpret_cast<const uint4*>(&x[k]);
            
            half2* a_ptr = reinterpret_cast<half2*>(&a_vec);
            half2* x_ptr = reinterpret_cast<half2*>(&x_vec);
            
            for (int i = 0; i < 4; i++) {
                sum += __half2float(a_ptr[i].x) * __half2float(x_ptr[i].x);
                sum += __half2float(a_ptr[i].y) * __half2float(x_ptr[i].y);
            }
        }
        
        // Handle remaining elements
        for (; k < K; k++) {
            sum += __half2float(a_row[k]) * __half2float(x[k]);
        }
        
        y[row] = __float2half(sum);
    }
}

// ----------------------------------------------------------------------------
// Dispatcher Kernel (exported as cuda_kernel)
// ----------------------------------------------------------------------------

extern "C" void cuda_kernel(const void* A, const void* x, void* y, 
                            int M, int K, int op_type) {
    // Thread block size
    const int BLOCK_SIZE = 256;
    
    // Grid size
    int num_blocks = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    switch (op_type) {
        case 0: // FP32, K=16
            gemv_k16_fp32<<<num_blocks, BLOCK_SIZE>>>(
                static_cast<const float*>(A),
                static_cast<const float*>(x),
                static_cast<float*>(y),
                M, K
            );
            break;
            
        case 1: // FP32, K=32/64 general
            gemv_k32_fp32<<<num_blocks, BLOCK_SIZE>>>(
                static_cast<const float*>(A),
                static_cast<const float*>(x),
                static_cast<float*>(y),
                M, K
            );
            break;
            
        case 2: // FP32, K=128 vectorized
            gemv_k128_fp32<<<num_blocks, BLOCK_SIZE>>>(
                static_cast<const float*>(A),
                static_cast<const float*>(x),
                static_cast<float*>(y),
                M, K
            );
            break;
            
        case 3: // FP16, K=16
            gemv_k16_fp16<<<num_blocks, BLOCK_SIZE>>>(
                static_cast<const half*>(A),
                static_cast<const half*>(x),
                static_cast<half*>(y),
                M, K
            );
            break;
            
        case 4: // FP16, K=32/64 general
            gemv_k32_fp16<<<num_blocks, BLOCK_SIZE>>>(
                static_cast<const half*>(A),
                static_cast<const half*>(x),
                static_cast<half*>(y),
                M, K
            );
            break;
            
        case 5: // FP16, K=128 vectorized
            gemv_k128_fp16<<<num_blocks, BLOCK_SIZE>>>(
                static_cast<const half*>(A),
                static_cast<const half*>(x),
                static_cast<half*>(y),
                M, K
            );
            break;
            
        default:
            // Use general kernel as fallback
            if (op_type < 3) {
                gemv_k32_fp32<<<num_blocks, BLOCK_SIZE>>>(
                    static_cast<const float*>(A),
                    static_cast<const float*>(x),
                    static_cast<float*>(y),
                    M, K
                );
            } else {
                gemv_k32_fp16<<<num_blocks, BLOCK_SIZE>>>(
                    static_cast<const half*>(A),
                    static_cast<const half*>(x),
                    static_cast<half*>(y),
                    M, K
                );
            }
            break;
    }
}
