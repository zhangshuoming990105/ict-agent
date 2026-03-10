#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdint>

// FP8 type definitions (if not available in cuda_fp8.h)
#if !defined(__CUDA_FP8_TYPES_EXIST__)
struct __nv_fp8_e4m3 {
    uint8_t __x;
    __device__ __host__ __nv_fp8_e4m3() = default;
    __device__ __host__ explicit operator float() const {
        // Simple conversion for FP8 E4M3
        uint8_t x = __x;
        if (x == 0) return 0.0f;
        int sign = (x & 0x80) ? -1 : 1;
        int exp = (x >> 3) & 0x0F;
        int mant = x & 0x07;
        if (exp == 0) {
            // Subnormal
            return sign * (mant / 8.0f) * powf(2.0f, -6.0f);
        } else if (exp == 15) {
            // NaN
            return sign * INFINITY;
        } else {
            // Normal
            return sign * (1.0f + mant / 8.0f) * powf(2.0f, exp - 7.0f);
        }
    }
};

struct __nv_fp8_e5m2 {
    uint8_t __x;
    __device__ __host__ __nv_fp8_e5m2() = default;
    __device__ __host__ explicit operator float() const {
        // Simple conversion for FP8 E5M2
        uint8_t x = __x;
        if (x == 0) return 0.0f;
        int sign = (x & 0x80) ? -1 : 1;
        int exp = (x >> 2) & 0x1F;
        int mant = x & 0x03;
        if (exp == 0) {
            // Subnormal
            return sign * (mant / 4.0f) * powf(2.0f, -14.0f);
        } else if (exp == 31) {
            // NaN or Inf
            return sign * INFINITY;
        } else {
            // Normal
            return sign * (1.0f + mant / 4.0f) * powf(2.0f, exp - 15.0f);
        }
    }
};
#endif

// Block reduce for float
__device__ __forceinline__ float blockReduceSum(float val) {
    __shared__ float shared[32];
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;
    
    // Warp reduce
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    
    // Write warp result to shared memory
    if (lane == 0) shared[wid] = val;
    __syncthreads();
    
    // Final reduce in first warp
    if (wid == 0) {
        val = (threadIdx.x < blockDim.x / 32) ? shared[lane] : 0.0f;
        for (int offset = 16; offset > 0; offset /= 2) {
            val += __shfl_down_sync(0xffffffff, val, offset);
        }
    }
    
    return val;
}

// Block reduce for int
__device__ __forceinline__ int blockReduceSumInt(int val) {
    __shared__ int shared[32];
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;
    
    // Warp reduce
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    
    // Write warp result to shared memory
    if (lane == 0) shared[wid] = val;
    __syncthreads();
    
    // Final reduce in first warp
    if (wid == 0) {
        val = (threadIdx.x < blockDim.x / 32) ? shared[lane] : 0;
        for (int offset = 16; offset > 0; offset /= 2) {
            val += __shfl_down_sync(0xffffffff, val, offset);
        }
    }
    
    return val;
}

// ==================== F32 Kernels ====================
// op_type = 0: f32 -> f32
__global__ void reduce_sum_f32_f32(const float* input, float* output, int n) {
    float sum = 0.0f;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    for (int i = idx; i < n; i += stride) {
        sum += input[i];
    }
    
    sum = blockReduceSum(sum);
    
    if (threadIdx.x == 0) {
        atomicAdd(output, sum);
    }
}

// op_type = 1: f32x4 -> f32
__global__ void reduce_sum_f32x4_f32(const float* input, float* output, int n) {
    float sum = 0.0f;
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    int stride = blockDim.x * gridDim.x * 4;
    
    for (int i = idx; i < n; i += stride) {
        if (i + 3 < n) {
            float4 val = *reinterpret_cast<const float4*>(&input[i]);
            sum += val.x + val.y + val.z + val.w;
        } else {
            for (int j = i; j < n && j < i + 4; j++) {
                sum += input[j];
            }
        }
    }
    
    sum = blockReduceSum(sum);
    
    if (threadIdx.x == 0) {
        atomicAdd(output, sum);
    }
}

// ==================== F16 Kernels ====================
// op_type = 2: f16 -> f16 (accumulate in f16)
__global__ void reduce_sum_f16_f16(const __half* input, float* output, int n) {
    float sum = 0.0f;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    for (int i = idx; i < n; i += stride) {
        sum += __half2float(input[i]);
    }
    
    sum = blockReduceSum(sum);
    
    if (threadIdx.x == 0) {
        atomicAdd(output, sum);
    }
}

// op_type = 3: f16 -> f32
__global__ void reduce_sum_f16_f32(const __half* input, float* output, int n) {
    float sum = 0.0f;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    for (int i = idx; i < n; i += stride) {
        sum += __half2float(input[i]);
    }
    
    sum = blockReduceSum(sum);
    
    if (threadIdx.x == 0) {
        atomicAdd(output, sum);
    }
}

// op_type = 4: f16x2 -> f16 (accumulate in f16)
__global__ void reduce_sum_f16x2_f16(const __half* input, float* output, int n) {
    float sum = 0.0f;
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
    int stride = blockDim.x * gridDim.x * 2;
    
    for (int i = idx; i < n; i += stride) {
        if (i + 1 < n) {
            __half2 val = *reinterpret_cast<const __half2*>(&input[i]);
            sum += __half2float(val.x) + __half2float(val.y);
        } else if (i < n) {
            sum += __half2float(input[i]);
        }
    }
    
    sum = blockReduceSum(sum);
    
    if (threadIdx.x == 0) {
        atomicAdd(output, sum);
    }
}

// op_type = 5: f16x2 -> f32
__global__ void reduce_sum_f16x2_f32(const __half* input, float* output, int n) {
    float sum = 0.0f;
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
    int stride = blockDim.x * gridDim.x * 2;
    
    for (int i = idx; i < n; i += stride) {
        if (i + 1 < n) {
            __half2 val = *reinterpret_cast<const __half2*>(&input[i]);
            sum += __half2float(val.x) + __half2float(val.y);
        } else if (i < n) {
            sum += __half2float(input[i]);
        }
    }
    
    sum = blockReduceSum(sum);
    
    if (threadIdx.x == 0) {
        atomicAdd(output, sum);
    }
}

// op_type = 6: f16x8_pack -> f16 (accumulate in f16)
__global__ void reduce_sum_f16x8_pack_f16(const __half* input, float* output, int n) {
    float sum = 0.0f;
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 8;
    int stride = blockDim.x * gridDim.x * 8;
    
    for (int i = idx; i < n; i += stride) {
        for (int j = 0; j < 8 && i + j < n; j++) {
            sum += __half2float(input[i + j]);
        }
    }
    
    sum = blockReduceSum(sum);
    
    if (threadIdx.x == 0) {
        atomicAdd(output, sum);
    }
}

// op_type = 7: f16x8_pack -> f32
__global__ void reduce_sum_f16x8_pack_f32(const __half* input, float* output, int n) {
    float sum = 0.0f;
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 8;
    int stride = blockDim.x * gridDim.x * 8;
    
    for (int i = idx; i < n; i += stride) {
        for (int j = 0; j < 8 && i + j < n; j++) {
            sum += __half2float(input[i + j]);
        }
    }
    
    sum = blockReduceSum(sum);
    
    if (threadIdx.x == 0) {
        atomicAdd(output, sum);
    }
}

// ==================== BF16 Kernels ====================
// op_type = 8: bf16 -> bf16 (accumulate in bf16)
__global__ void reduce_sum_bf16_bf16(const __nv_bfloat16* input, float* output, int n) {
    float sum = 0.0f;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    for (int i = idx; i < n; i += stride) {
        sum += __bfloat162float(input[i]);
    }
    
    sum = blockReduceSum(sum);
    
    if (threadIdx.x == 0) {
        atomicAdd(output, sum);
    }
}

// op_type = 9: bf16 -> f32
__global__ void reduce_sum_bf16_f32(const __nv_bfloat16* input, float* output, int n) {
    float sum = 0.0f;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    for (int i = idx; i < n; i += stride) {
        sum += __bfloat162float(input[i]);
    }
    
    sum = blockReduceSum(sum);
    
    if (threadIdx.x == 0) {
        atomicAdd(output, sum);
    }
}

// op_type = 10: bf16x2 -> bf16 (accumulate in bf16)
__global__ void reduce_sum_bf16x2_bf16(const __nv_bfloat16* input, float* output, int n) {
    float sum = 0.0f;
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
    int stride = blockDim.x * gridDim.x * 2;
    
    for (int i = idx; i < n; i += stride) {
        if (i + 1 < n) {
            __nv_bfloat162 val = *reinterpret_cast<const __nv_bfloat162*>(&input[i]);
            sum += __bfloat162float(val.x) + __bfloat162float(val.y);
        } else if (i < n) {
            sum += __bfloat162float(input[i]);
        }
    }
    
    sum = blockReduceSum(sum);
    
    if (threadIdx.x == 0) {
        atomicAdd(output, sum);
    }
}

// op_type = 11: bf16x2 -> f32
__global__ void reduce_sum_bf16x2_f32(const __nv_bfloat16* input, float* output, int n) {
    float sum = 0.0f;
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
    int stride = blockDim.x * gridDim.x * 2;
    
    for (int i = idx; i < n; i += stride) {
        if (i + 1 < n) {
            __nv_bfloat162 val = *reinterpret_cast<const __nv_bfloat162*>(&input[i]);
            sum += __bfloat162float(val.x) + __bfloat162float(val.y);
        } else if (i < n) {
            sum += __bfloat162float(input[i]);
        }
    }
    
    sum = blockReduceSum(sum);
    
    if (threadIdx.x == 0) {
        atomicAdd(output, sum);
    }
}

// op_type = 12: bf16x8_pack -> bf16 (accumulate in bf16)
__global__ void reduce_sum_bf16x8_pack_bf16(const __nv_bfloat16* input, float* output, int n) {
    float sum = 0.0f;
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 8;
    int stride = blockDim.x * gridDim.x * 8;
    
    for (int i = idx; i < n; i += stride) {
        for (int j = 0; j < 8 && i + j < n; j++) {
            sum += __bfloat162float(input[i + j]);
        }
    }
    
    sum = blockReduceSum(sum);
    
    if (threadIdx.x == 0) {
        atomicAdd(output, sum);
    }
}

// op_type = 13: bf16x8_pack -> f32
__global__ void reduce_sum_bf16x8_pack_f32(const __nv_bfloat16* input, float* output, int n) {
    float sum = 0.0f;
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 8;
    int stride = blockDim.x * gridDim.x * 8;
    
    for (int i = idx; i < n; i += stride) {
        for (int j = 0; j < 8 && i + j < n; j++) {
            sum += __bfloat162float(input[i + j]);
        }
    }
    
    sum = blockReduceSum(sum);
    
    if (threadIdx.x == 0) {
        atomicAdd(output, sum);
    }
}

// ==================== FP8 Kernels ====================
// op_type = 14: fp8_e4m3 -> f16
__global__ void reduce_sum_fp8_e4m3_f16(const __nv_fp8_e4m3* input, float* output, int n) {
    float sum = 0.0f;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    for (int i = idx; i < n; i += stride) {
        sum += float(input[i]);
    }
    
    sum = blockReduceSum(sum);
    
    if (threadIdx.x == 0) {
        atomicAdd(output, sum);
    }
}

// op_type = 15: fp8_e4m3x16_pack -> f16
__global__ void reduce_sum_fp8_e4m3x16_pack_f16(const __nv_fp8_e4m3* input, float* output, int n) {
    float sum = 0.0f;
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 16;
    int stride = blockDim.x * gridDim.x * 16;
    
    for (int i = idx; i < n; i += stride) {
        for (int j = 0; j < 16 && i + j < n; j++) {
            sum += float(input[i + j]);
        }
    }
    
    sum = blockReduceSum(sum);
    
    if (threadIdx.x == 0) {
        atomicAdd(output, sum);
    }
}

// op_type = 16: fp8_e5m2 -> f16
__global__ void reduce_sum_fp8_e5m2_f16(const __nv_fp8_e5m2* input, float* output, int n) {
    float sum = 0.0f;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    for (int i = idx; i < n; i += stride) {
        sum += float(input[i]);
    }
    
    sum = blockReduceSum(sum);
    
    if (threadIdx.x == 0) {
        atomicAdd(output, sum);
    }
}

// op_type = 17: fp8_e5m2x16_pack -> f16
__global__ void reduce_sum_fp8_e5m2x16_pack_f16(const __nv_fp8_e5m2* input, float* output, int n) {
    float sum = 0.0f;
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 16;
    int stride = blockDim.x * gridDim.x * 16;
    
    for (int i = idx; i < n; i += stride) {
        for (int j = 0; j < 16 && i + j < n; j++) {
            sum += float(input[i + j]);
        }
    }
    
    sum = blockReduceSum(sum);
    
    if (threadIdx.x == 0) {
        atomicAdd(output, sum);
    }
}

// ==================== INT8 Kernels ====================
// op_type = 18: i8 -> i32
__global__ void reduce_sum_i8_i32(const int8_t* input, int* output, int n) {
    int sum = 0;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    for (int i = idx; i < n; i += stride) {
        sum += (int)input[i];
    }
    
    sum = blockReduceSumInt(sum);
    
    if (threadIdx.x == 0) {
        atomicAdd(output, sum);
    }
}

// op_type = 19: i8x16_pack -> i32
__global__ void reduce_sum_i8x16_pack_i32(const int8_t* input, int* output, int n) {
    int sum = 0;
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 16;
    int stride = blockDim.x * gridDim.x * 16;
    
    for (int i = idx; i < n; i += stride) {
        for (int j = 0; j < 16 && i + j < n; j++) {
            sum += (int)input[i + j];
        }
    }
    
    sum = blockReduceSumInt(sum);
    
    if (threadIdx.x == 0) {
        atomicAdd(output, sum);
    }
}

// ==================== Main Kernel Dispatcher ====================
extern "C" void cuda_kernel(void* input, void* output, int n, int op_type) {
    int threads = 256;
    int blocks = min((n + threads - 1) / threads, 1024);
    
    switch(op_type) {
        case 0:
            reduce_sum_f32_f32<<<blocks, threads>>>((float*)input, (float*)output, n);
            break;
        case 1:
            reduce_sum_f32x4_f32<<<blocks, threads>>>((float*)input, (float*)output, n);
            break;
        case 2:
            reduce_sum_f16_f16<<<blocks, threads>>>((__half*)input, (float*)output, n);
            break;
        case 3:
            reduce_sum_f16_f32<<<blocks, threads>>>((__half*)input, (float*)output, n);
            break;
        case 4:
            reduce_sum_f16x2_f16<<<blocks, threads>>>((__half*)input, (float*)output, n);
            break;
        case 5:
            reduce_sum_f16x2_f32<<<blocks, threads>>>((__half*)input, (float*)output, n);
            break;
        case 6:
            reduce_sum_f16x8_pack_f16<<<blocks, threads>>>((__half*)input, (float*)output, n);
            break;
        case 7:
            reduce_sum_f16x8_pack_f32<<<blocks, threads>>>((__half*)input, (float*)output, n);
            break;
        case 8:
            reduce_sum_bf16_bf16<<<blocks, threads>>>((__nv_bfloat16*)input, (float*)output, n);
            break;
        case 9:
            reduce_sum_bf16_f32<<<blocks, threads>>>((__nv_bfloat16*)input, (float*)output, n);
            break;
        case 10:
            reduce_sum_bf16x2_bf16<<<blocks, threads>>>((__nv_bfloat16*)input, (float*)output, n);
            break;
        case 11:
            reduce_sum_bf16x2_f32<<<blocks, threads>>>((__nv_bfloat16*)input, (float*)output, n);
            break;
        case 12:
            reduce_sum_bf16x8_pack_bf16<<<blocks, threads>>>((__nv_bfloat16*)input, (float*)output, n);
            break;
        case 13:
            reduce_sum_bf16x8_pack_f32<<<blocks, threads>>>((__nv_bfloat16*)input, (float*)output, n);
            break;
        case 14:
            reduce_sum_fp8_e4m3_f16<<<blocks, threads>>>((__nv_fp8_e4m3*)input, (float*)output, n);
            break;
        case 15:
            reduce_sum_fp8_e4m3x16_pack_f16<<<blocks, threads>>>((__nv_fp8_e4m3*)input, (float*)output, n);
            break;
        case 16:
            reduce_sum_fp8_e5m2_f16<<<blocks, threads>>>((__nv_fp8_e5m2*)input, (float*)output, n);
            break;
        case 17:
            reduce_sum_fp8_e5m2x16_pack_f16<<<blocks, threads>>>((__nv_fp8_e5m2*)input, (float*)output, n);
            break;
        case 18:
            reduce_sum_i8_i32<<<blocks, threads>>>((int8_t*)input, (int*)output, n);
            break;
        case 19:
            reduce_sum_i8x16_pack_i32<<<blocks, threads>>>((int8_t*)input, (int*)output, n);
            break;
        default:
            // Invalid op_type
            break;
    }
}
