#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <float.h>

// Warp-level reduction for maximum
template <typename T>
__device__ __forceinline__ T warpReduceMax(T val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val = max(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

// Warp-level reduction for sum
template <typename T>
__device__ __forceinline__ T warpReduceSum(T val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Softmax kernel for float32
__global__ void softmax_f32_kernel(const float* __restrict__ input,
                                    float* __restrict__ output,
                                    int S, int H) {
    int row = blockIdx.x;
    if (row >= S) return;
    
    const float* x = input + row * H;
    float* y = output + row * H;
    
    __shared__ float shared_mem[32];
    
    // Find max value
    float max_val = -FLT_MAX;
    for (int i = threadIdx.x; i < H; i += blockDim.x) {
        max_val = max(max_val, x[i]);
    }
    max_val = warpReduceMax(max_val);
    if (threadIdx.x % 32 == 0) {
        shared_mem[threadIdx.x / 32] = max_val;
    }
    __syncthreads();
    
    if (threadIdx.x == 0) {
        max_val = shared_mem[0];
        for (int i = 1; i < (blockDim.x + 31) / 32; i++) {
            max_val = max(max_val, shared_mem[i]);
        }
        shared_mem[0] = max_val;
    }
    __syncthreads();
    max_val = shared_mem[0];
    
    // Compute exp and sum
    float sum_val = 0.0f;
    for (int i = threadIdx.x; i < H; i += blockDim.x) {
        float exp_val = expf(x[i] - max_val);
        y[i] = exp_val;
        sum_val += exp_val;
    }
    sum_val = warpReduceSum(sum_val);
    if (threadIdx.x % 32 == 0) {
        shared_mem[threadIdx.x / 32] = sum_val;
    }
    __syncthreads();
    
    if (threadIdx.x == 0) {
        sum_val = shared_mem[0];
        for (int i = 1; i < (blockDim.x + 31) / 32; i++) {
            sum_val += shared_mem[i];
        }
        shared_mem[0] = sum_val;
    }
    __syncthreads();
    sum_val = shared_mem[0];
    
    // Normalize
    for (int i = threadIdx.x; i < H; i += blockDim.x) {
        y[i] = y[i] / sum_val;
    }
}

// Softmax kernel for float16
__global__ void softmax_f16_kernel(const __half* __restrict__ input,
                                    __half* __restrict__ output,
                                    int S, int H) {
    int row = blockIdx.x;
    if (row >= S) return;
    
    const __half* x = input + row * H;
    __half* y = output + row * H;
    
    __shared__ float shared_mem[32];
    
    // Find max value (compute in float for stability)
    float max_val = -FLT_MAX;
    for (int i = threadIdx.x; i < H; i += blockDim.x) {
        max_val = max(max_val, __half2float(x[i]));
    }
    max_val = warpReduceMax(max_val);
    if (threadIdx.x % 32 == 0) {
        shared_mem[threadIdx.x / 32] = max_val;
    }
    __syncthreads();
    
    if (threadIdx.x == 0) {
        max_val = shared_mem[0];
        for (int i = 1; i < (blockDim.x + 31) / 32; i++) {
            max_val = max(max_val, shared_mem[i]);
        }
        shared_mem[0] = max_val;
    }
    __syncthreads();
    max_val = shared_mem[0];
    
    // Compute exp and sum
    float sum_val = 0.0f;
    for (int i = threadIdx.x; i < H; i += blockDim.x) {
        float exp_val = expf(__half2float(x[i]) - max_val);
        y[i] = __float2half(exp_val);
        sum_val += exp_val;
    }
    sum_val = warpReduceSum(sum_val);
    if (threadIdx.x % 32 == 0) {
        shared_mem[threadIdx.x / 32] = sum_val;
    }
    __syncthreads();
    
    if (threadIdx.x == 0) {
        sum_val = shared_mem[0];
        for (int i = 1; i < (blockDim.x + 31) / 32; i++) {
            sum_val += shared_mem[i];
        }
        shared_mem[0] = sum_val;
    }
    __syncthreads();
    sum_val = shared_mem[0];
    
    // Normalize
    for (int i = threadIdx.x; i < H; i += blockDim.x) {
        y[i] = __float2half(__half2float(y[i]) / sum_val);
    }
}

// C interface expected by the operator test framework
extern "C" {
    void cuda_kernel(void* input, void* output, int S, int H, int op_type) {
        dim3 grid(S);
        // Adaptive block size based on problem size
        int block_size = (H <= 256) ? 128 : (H <= 1024) ? 256 : 512;
        dim3 block(block_size);
        
        // Determine dtype by op_type
        // F32: 0,1,2,3,7,8
        // F16: 4,5,6
        bool use_f16 = (op_type >= 4 && op_type <= 6);
        
        if (use_f16) {
            softmax_f16_kernel<<<grid, block>>>(
                reinterpret_cast<const __half*>(input),
                reinterpret_cast<__half*>(output),
                S, H
            );
        } else {
            softmax_f32_kernel<<<grid, block>>>(
                reinterpret_cast<const float*>(input),
                reinterpret_cast<float*>(output),
                S, H
            );
        }
    }
}
