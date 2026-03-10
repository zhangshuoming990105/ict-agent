#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdint>

// FP8 manual conversion helpers for CUDA < 11.8
// E4M3 format: 1 sign bit, 4 exponent bits, 3 mantissa bits
__device__ __forceinline__ float fp8_e4m3_to_float(uint8_t x) {
    uint32_t sign = (x & 0x80) << 24;
    uint32_t exp = (x & 0x78) >> 3;
    uint32_t mantissa = (x & 0x07);
    
    if (exp == 0) {
        if (mantissa == 0) return __int_as_float(sign); // zero
        // Subnormal
        float val = mantissa * 0.001953125f; // 2^-9
        return __int_as_float(sign) < 0 ? -val : val;
    }
    if (exp == 15) { // NaN
        return __int_as_float(0x7fc00000);
    }
    
    uint32_t exp_bits = (exp - 7 + 127) << 23;
    uint32_t mantissa_bits = mantissa << 20;
    return __int_as_float(sign | exp_bits | mantissa_bits);
}

// E5M2 format: 1 sign bit, 5 exponent bits, 2 mantissa bits
__device__ __forceinline__ float fp8_e5m2_to_float(uint8_t x) {
    uint32_t sign = (x & 0x80) << 24;
    uint32_t exp = (x & 0x7C) >> 2;
    uint32_t mantissa = (x & 0x03);
    
    if (exp == 0) {
        if (mantissa == 0) return __int_as_float(sign); // zero
        // Subnormal
        float val = mantissa * 0.00024414062f; // 2^-12
        return __int_as_float(sign) < 0 ? -val : val;
    }
    if (exp == 31) { // Inf or NaN
        if (mantissa == 0) return __int_as_float(sign | 0x7f800000); // Inf
        return __int_as_float(0x7fc00000); // NaN
    }
    
    uint32_t exp_bits = (exp - 15 + 127) << 23;
    uint32_t mantissa_bits = mantissa << 21;
    return __int_as_float(sign | exp_bits | mantissa_bits);
}

// FP8 types compatible with PyTorch's storage
struct __nv_fp8_e4m3 { 
    uint8_t __x; 
    __device__ __forceinline__ operator float() const { return fp8_e4m3_to_float(__x); }
};

struct __nv_fp8_e5m2 { 
    uint8_t __x; 
    __device__ __forceinline__ operator float() const { return fp8_e5m2_to_float(__x); }
};

// Warp-level reduction using shuffle
template<typename T>
__inline__ __device__ T warpReduce(T val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Block-level reduction
template<typename T>
__inline__ __device__ T blockReduce(T val) {
    __shared__ T shared[32];
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;
    
    val = warpReduce(val);
    
    if (lane == 0) shared[wid] = val;
    __syncthreads();
    
    val = (threadIdx.x < blockDim.x / 32) ? shared[lane] : T(0);
    if (wid == 0) val = warpReduce(val);
    
    return val;
}

// FP32 scalar reduction
__global__ void reduce_f32_kernel(const float* input, float* output, int n) {
    float sum = 0.0f;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        sum += input[i];
    }
    sum = blockReduce(sum);
    if (threadIdx.x == 0) {
        atomicAdd(output, sum);
    }
}

// FP32 vectorized (float4) reduction
__global__ void reduce_f32x4_kernel(const float4* input, float* output, int n) {
    float sum = 0.0f;
    int n4 = n / 4;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += blockDim.x * gridDim.x) {
        float4 val = input[i];
        sum += val.x + val.y + val.z + val.w;
    }
    sum = blockReduce(sum);
    if (threadIdx.x == 0) {
        atomicAdd(output, sum);
    }
}

// FP16 scalar with FP16 accumulator
__global__ void reduce_f16_f16_kernel(const __half* input, float* output, int n) {
    __half sum = __float2half(0.0f);
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        sum = __hadd(sum, input[i]);
    }
    float sum_f32 = __half2float(sum);
    sum_f32 = blockReduce(sum_f32);
    if (threadIdx.x == 0) {
        atomicAdd(output, sum_f32);
    }
}

// FP16 scalar with FP32 accumulator
__global__ void reduce_f16_f32_kernel(const __half* input, float* output, int n) {
    float sum = 0.0f;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        sum += __half2float(input[i]);
    }
    sum = blockReduce(sum);
    if (threadIdx.x == 0) {
        atomicAdd(output, sum);
    }
}

// FP16 vectorized (half2) with FP16 accumulator
__global__ void reduce_f16x2_f16_kernel(const __half2* input, float* output, int n) {
    __half2 sum = __float2half2_rn(0.0f);
    int n2 = n / 2;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n2; i += blockDim.x * gridDim.x) {
        sum = __hadd2(sum, input[i]);
    }
    __half sum_low = __low2half(sum);
    __half sum_high = __high2half(sum);
    float sum_f32 = __half2float(__hadd(sum_low, sum_high));
    sum_f32 = blockReduce(sum_f32);
    if (threadIdx.x == 0) {
        atomicAdd(output, sum_f32);
    }
}

// FP16 vectorized (half2) with FP32 accumulator
__global__ void reduce_f16x2_f32_kernel(const __half2* input, float* output, int n) {
    float sum = 0.0f;
    int n2 = n / 2;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n2; i += blockDim.x * gridDim.x) {
        __half2 val = input[i];
        sum += __half2float(__low2half(val)) + __half2float(__high2half(val));
    }
    sum = blockReduce(sum);
    if (threadIdx.x == 0) {
        atomicAdd(output, sum);
    }
}

// FP16 vectorized (8 halfs packed) with FP16 accumulator
__global__ void reduce_f16x8_pack_f16_kernel(const uint4* input, float* output, int n) {
    __half sum = __float2half(0.0f);
    int n8 = n / 8;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n8; i += blockDim.x * gridDim.x) {
        uint4 packed = input[i];
        __half2* h2_ptr = reinterpret_cast<__half2*>(&packed);
        for (int j = 0; j < 4; j++) {
            sum = __hadd(sum, __hadd(__low2half(h2_ptr[j]), __high2half(h2_ptr[j])));
        }
    }
    float sum_f32 = __half2float(sum);
    sum_f32 = blockReduce(sum_f32);
    if (threadIdx.x == 0) {
        atomicAdd(output, sum_f32);
    }
}

// FP16 vectorized (8 halfs packed) with FP32 accumulator
__global__ void reduce_f16x8_pack_f32_kernel(const uint4* input, float* output, int n) {
    float sum = 0.0f;
    int n8 = n / 8;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n8; i += blockDim.x * gridDim.x) {
        uint4 packed = input[i];
        __half2* h2_ptr = reinterpret_cast<__half2*>(&packed);
        for (int j = 0; j < 4; j++) {
            sum += __half2float(__low2half(h2_ptr[j])) + __half2float(__high2half(h2_ptr[j]));
        }
    }
    sum = blockReduce(sum);
    if (threadIdx.x == 0) {
        atomicAdd(output, sum);
    }
}

// BF16 scalar with BF16 accumulator
__global__ void reduce_bf16_bf16_kernel(const __nv_bfloat16* input, float* output, int n) {
    __nv_bfloat16 sum = __float2bfloat16(0.0f);
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        sum = __hadd(sum, input[i]);
    }
    float sum_f32 = __bfloat162float(sum);
    sum_f32 = blockReduce(sum_f32);
    if (threadIdx.x == 0) {
        atomicAdd(output, sum_f32);
    }
}

// BF16 scalar with FP32 accumulator
__global__ void reduce_bf16_f32_kernel(const __nv_bfloat16* input, float* output, int n) {
    float sum = 0.0f;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        sum += __bfloat162float(input[i]);
    }
    sum = blockReduce(sum);
    if (threadIdx.x == 0) {
        atomicAdd(output, sum);
    }
}

// BF16 vectorized (bfloat162) with BF16 accumulator
__global__ void reduce_bf16x2_bf16_kernel(const __nv_bfloat162* input, float* output, int n) {
    __nv_bfloat162 sum = __float2bfloat162_rn(0.0f);
    int n2 = n / 2;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n2; i += blockDim.x * gridDim.x) {
        sum = __hadd2(sum, input[i]);
    }
    __nv_bfloat16 sum_low = __low2bfloat16(sum);
    __nv_bfloat16 sum_high = __high2bfloat16(sum);
    float sum_f32 = __bfloat162float(__hadd(sum_low, sum_high));
    sum_f32 = blockReduce(sum_f32);
    if (threadIdx.x == 0) {
        atomicAdd(output, sum_f32);
    }
}

// BF16 vectorized (bfloat162) with FP32 accumulator
__global__ void reduce_bf16x2_f32_kernel(const __nv_bfloat162* input, float* output, int n) {
    float sum = 0.0f;
    int n2 = n / 2;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n2; i += blockDim.x * gridDim.x) {
        __nv_bfloat162 val = input[i];
        sum += __bfloat162float(__low2bfloat16(val)) + __bfloat162float(__high2bfloat16(val));
    }
    sum = blockReduce(sum);
    if (threadIdx.x == 0) {
        atomicAdd(output, sum);
    }
}

// BF16 vectorized (8 bfloat16s packed) with BF16 accumulator
__global__ void reduce_bf16x8_pack_bf16_kernel(const uint4* input, float* output, int n) {
    __nv_bfloat16 sum = __float2bfloat16(0.0f);
    int n8 = n / 8;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n8; i += blockDim.x * gridDim.x) {
        uint4 packed = input[i];
        __nv_bfloat162* b2_ptr = reinterpret_cast<__nv_bfloat162*>(&packed);
        for (int j = 0; j < 4; j++) {
            sum = __hadd(sum, __hadd(__low2bfloat16(b2_ptr[j]), __high2bfloat16(b2_ptr[j])));
        }
    }
    float sum_f32 = __bfloat162float(sum);
    sum_f32 = blockReduce(sum_f32);
    if (threadIdx.x == 0) {
        atomicAdd(output, sum_f32);
    }
}

// BF16 vectorized (8 bfloat16s packed) with FP32 accumulator
__global__ void reduce_bf16x8_pack_f32_kernel(const uint4* input, float* output, int n) {
    float sum = 0.0f;
    int n8 = n / 8;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n8; i += blockDim.x * gridDim.x) {
        uint4 packed = input[i];
        __nv_bfloat162* b2_ptr = reinterpret_cast<__nv_bfloat162*>(&packed);
        for (int j = 0; j < 4; j++) {
            sum += __bfloat162float(__low2bfloat16(b2_ptr[j])) + __bfloat162float(__high2bfloat16(b2_ptr[j]));
        }
    }
    sum = blockReduce(sum);
    if (threadIdx.x == 0) {
        atomicAdd(output, sum);
    }
}

// FP8 E4M3 scalar reduction
__global__ void reduce_fp8_e4m3_kernel(const __nv_fp8_e4m3* input, float* output, int n) {
    float sum = 0.0f;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        sum += float(input[i]);
    }
    sum = blockReduce(sum);
    if (threadIdx.x == 0) {
        atomicAdd(output, sum);
    }
}

// FP8 E4M3 vectorized (16 packed) reduction
__global__ void reduce_fp8_e4m3x16_pack_kernel(const uint4* input, float* output, int n) {
    float sum = 0.0f;
    int n16 = n / 16;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n16; i += blockDim.x * gridDim.x) {
        uint4 packed = input[i];
        __nv_fp8_e4m3* fp8_ptr = reinterpret_cast<__nv_fp8_e4m3*>(&packed);
        for (int j = 0; j < 16; j++) {
            sum += float(fp8_ptr[j]);
        }
    }
    sum = blockReduce(sum);
    if (threadIdx.x == 0) {
        atomicAdd(output, sum);
    }
}

// FP8 E5M2 scalar reduction
__global__ void reduce_fp8_e5m2_kernel(const __nv_fp8_e5m2* input, float* output, int n) {
    float sum = 0.0f;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        sum += float(input[i]);
    }
    sum = blockReduce(sum);
    if (threadIdx.x == 0) {
        atomicAdd(output, sum);
    }
}

// FP8 E5M2 vectorized (16 packed) reduction
__global__ void reduce_fp8_e5m2x16_pack_kernel(const uint4* input, float* output, int n) {
    float sum = 0.0f;
    int n16 = n / 16;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n16; i += blockDim.x * gridDim.x) {
        uint4 packed = input[i];
        __nv_fp8_e5m2* fp8_ptr = reinterpret_cast<__nv_fp8_e5m2*>(&packed);
        for (int j = 0; j < 16; j++) {
            sum += float(fp8_ptr[j]);
        }
    }
    sum = blockReduce(sum);
    if (threadIdx.x == 0) {
        atomicAdd(output, sum);
    }
}

// INT8 scalar reduction
__global__ void reduce_i8_i32_kernel(const int8_t* input, int32_t* output, int n) {
    int32_t sum = 0;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        sum += static_cast<int32_t>(input[i]);
    }
    sum = blockReduce(sum);
    if (threadIdx.x == 0) {
        atomicAdd(output, sum);
    }
}

// INT8 vectorized (16 packed) reduction
__global__ void reduce_i8x16_pack_i32_kernel(const uint4* input, int32_t* output, int n) {
    int32_t sum = 0;
    int n16 = n / 16;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n16; i += blockDim.x * gridDim.x) {
        uint4 packed = input[i];
        int8_t* i8_ptr = reinterpret_cast<int8_t*>(&packed);
        for (int j = 0; j < 16; j++) {
            sum += static_cast<int32_t>(i8_ptr[j]);
        }
    }
    sum = blockReduce(sum);
    if (threadIdx.x == 0) {
        atomicAdd(output, sum);
    }
}

// Host wrapper function
extern "C" void cuda_kernel(void* input, void* output, int n, int op_type) {
    int threads = 256;
    int blocks = min((n + threads - 1) / threads, 1024);
    
    switch(op_type) {
        case 0: // f32_f32
            reduce_f32_kernel<<<blocks, threads>>>(
                reinterpret_cast<const float*>(input),
                reinterpret_cast<float*>(output),
                n);
            break;
        case 1: // f32x4_f32
            reduce_f32x4_kernel<<<blocks, threads>>>(
                reinterpret_cast<const float4*>(input),
                reinterpret_cast<float*>(output),
                n);
            break;
        case 2: // f16_f16
            reduce_f16_f16_kernel<<<blocks, threads>>>(
                reinterpret_cast<const __half*>(input),
                reinterpret_cast<float*>(output),
                n);
            break;
        case 3: // f16_f32
            reduce_f16_f32_kernel<<<blocks, threads>>>(
                reinterpret_cast<const __half*>(input),
                reinterpret_cast<float*>(output),
                n);
            break;
        case 4: // f16x2_f16
            reduce_f16x2_f16_kernel<<<blocks, threads>>>(
                reinterpret_cast<const __half2*>(input),
                reinterpret_cast<float*>(output),
                n);
            break;
        case 5: // f16x2_f32
            reduce_f16x2_f32_kernel<<<blocks, threads>>>(
                reinterpret_cast<const __half2*>(input),
                reinterpret_cast<float*>(output),
                n);
            break;
        case 6: // f16x8_pack_f16
            reduce_f16x8_pack_f16_kernel<<<blocks, threads>>>(
                reinterpret_cast<const uint4*>(input),
                reinterpret_cast<float*>(output),
                n);
            break;
        case 7: // f16x8_pack_f32
            reduce_f16x8_pack_f32_kernel<<<blocks, threads>>>(
                reinterpret_cast<const uint4*>(input),
                reinterpret_cast<float*>(output),
                n);
            break;
        case 8: // bf16_bf16
            reduce_bf16_bf16_kernel<<<blocks, threads>>>(
                reinterpret_cast<const __nv_bfloat16*>(input),
                reinterpret_cast<float*>(output),
                n);
            break;
        case 9: // bf16_f32
            reduce_bf16_f32_kernel<<<blocks, threads>>>(
                reinterpret_cast<const __nv_bfloat16*>(input),
                reinterpret_cast<float*>(output),
                n);
            break;
        case 10: // bf16x2_bf16
            reduce_bf16x2_bf16_kernel<<<blocks, threads>>>(
                reinterpret_cast<const __nv_bfloat162*>(input),
                reinterpret_cast<float*>(output),
                n);
            break;
        case 11: // bf16x2_f32
            reduce_bf16x2_f32_kernel<<<blocks, threads>>>(
                reinterpret_cast<const __nv_bfloat162*>(input),
                reinterpret_cast<float*>(output),
                n);
            break;
        case 12: // bf16x8_pack_bf16
            reduce_bf16x8_pack_bf16_kernel<<<blocks, threads>>>(
                reinterpret_cast<const uint4*>(input),
                reinterpret_cast<float*>(output),
                n);
            break;
        case 13: // bf16x8_pack_f32
            reduce_bf16x8_pack_f32_kernel<<<blocks, threads>>>(
                reinterpret_cast<const uint4*>(input),
                reinterpret_cast<float*>(output),
                n);
            break;
        case 14: // fp8_e4m3_f16
            reduce_fp8_e4m3_kernel<<<blocks, threads>>>(
                reinterpret_cast<const __nv_fp8_e4m3*>(input),
                reinterpret_cast<float*>(output),
                n);
            break;
        case 15: // fp8_e4m3x16_pack_f16
            reduce_fp8_e4m3x16_pack_kernel<<<blocks, threads>>>(
                reinterpret_cast<const uint4*>(input),
                reinterpret_cast<float*>(output),
                n);
            break;
        case 16: // fp8_e5m2_f16
            reduce_fp8_e5m2_kernel<<<blocks, threads>>>(
                reinterpret_cast<const __nv_fp8_e5m2*>(input),
                reinterpret_cast<float*>(output),
                n);
            break;
        case 17: // fp8_e5m2x16_pack_f16
            reduce_fp8_e5m2x16_pack_kernel<<<blocks, threads>>>(
                reinterpret_cast<const uint4*>(input),
                reinterpret_cast<float*>(output),
                n);
            break;
        case 18: // i8_i32
            reduce_i8_i32_kernel<<<blocks, threads>>>(
                reinterpret_cast<const int8_t*>(input),
                reinterpret_cast<int32_t*>(output),
                n);
            break;
        case 19: // i8x16_pack_i32
            reduce_i8x16_pack_i32_kernel<<<blocks, threads>>>(
                reinterpret_cast<const uint4*>(input),
                reinterpret_cast<int32_t*>(output),
                n);
            break;
    }
}
