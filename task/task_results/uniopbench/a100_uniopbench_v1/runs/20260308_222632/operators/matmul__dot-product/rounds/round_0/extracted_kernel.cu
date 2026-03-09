#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

// Warp-level reduction using shuffle instructions
__device__ __forceinline__ float warpReduceSum(float sum) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
    return sum;
}

// Block-level reduction using shared memory
__device__ __forceinline__ float blockReduceSum(float sum) {
    __shared__ float warp_sums[32];
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;
    
    // First, reduce within each warp
    sum = warpReduceSum(sum);
    
    // Write reduced value to shared memory
    if (lane == 0) {
        warp_sums[wid] = sum;
    }
    __syncthreads();
    
    // Have the first warp reduce the warp sums
    if (wid == 0) {
        sum = (threadIdx.x < (blockDim.x + 31) / 32) ? warp_sums[lane] : 0.0f;
        sum = warpReduceSum(sum);
    }
    
    return sum;
}

// Kernel 0: Basic float32 dot product
__global__ void dot_product_f32(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ y,
    int n
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    float sum = 0.0f;
    for (int i = tid; i < n; i += stride) {
        sum += a[i] * b[i];
    }
    
    sum = blockReduceSum(sum);
    
    if (threadIdx.x == 0) {
        atomicAdd(y, sum);
    }
}

// Kernel 1: Vectorized float4 dot product
__global__ void dot_product_f32x4(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ y,
    int n
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    int n_vec = n / 4;
    const float4* a_vec = reinterpret_cast<const float4*>(a);
    const float4* b_vec = reinterpret_cast<const float4*>(b);
    
    float sum = 0.0f;
    
    // Vectorized portion
    for (int i = tid; i < n_vec; i += stride) {
        float4 av = a_vec[i];
        float4 bv = b_vec[i];
        sum += av.x * bv.x + av.y * bv.y + av.z * bv.z + av.w * bv.w;
    }
    
    // Handle remainder
    for (int i = n_vec * 4 + tid; i < n; i += stride) {
        sum += a[i] * b[i];
    }
    
    sum = blockReduceSum(sum);
    
    if (threadIdx.x == 0) {
        atomicAdd(y, sum);
    }
}

// Kernel 2: Basic float16 dot product with float32 accumulation
__global__ void dot_product_f16(
    const half* __restrict__ a,
    const half* __restrict__ b,
    float* __restrict__ y,
    int n
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    float sum = 0.0f;
    for (int i = tid; i < n; i += stride) {
        sum += __half2float(a[i]) * __half2float(b[i]);
    }
    
    sum = blockReduceSum(sum);
    
    if (threadIdx.x == 0) {
        atomicAdd(y, sum);
    }
}

// Kernel 3: Vectorized half2 dot product with float32 accumulation
__global__ void dot_product_f16x2(
    const half* __restrict__ a,
    const half* __restrict__ b,
    float* __restrict__ y,
    int n
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    int n_vec = n / 2;
    const half2* a_vec = reinterpret_cast<const half2*>(a);
    const half2* b_vec = reinterpret_cast<const half2*>(b);
    
    float sum = 0.0f;
    
    // Vectorized portion
    for (int i = tid; i < n_vec; i += stride) {
        half2 av = a_vec[i];
        half2 bv = b_vec[i];
        half2 prod = __hmul2(av, bv);
        sum += __half2float(prod.x) + __half2float(prod.y);
    }
    
    // Handle remainder
    for (int i = n_vec * 2 + tid; i < n; i += stride) {
        sum += __half2float(a[i]) * __half2float(b[i]);
    }
    
    sum = blockReduceSum(sum);
    
    if (threadIdx.x == 0) {
        atomicAdd(y, sum);
    }
}

// Kernel 4: Vectorized float16x8 using uint4 packing
__global__ void dot_product_f16x8(
    const half* __restrict__ a,
    const half* __restrict__ b,
    float* __restrict__ y,
    int n
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    int n_vec = n / 8;
    const uint4* a_vec = reinterpret_cast<const uint4*>(a);
    const uint4* b_vec = reinterpret_cast<const uint4*>(b);
    
    float sum = 0.0f;
    
    // Vectorized portion (8 halfs = 1 uint4)
    for (int i = tid; i < n_vec; i += stride) {
        uint4 av = a_vec[i];
        uint4 bv = b_vec[i];
        
        const half2* a_half2 = reinterpret_cast<const half2*>(&av);
        const half2* b_half2 = reinterpret_cast<const half2*>(&bv);
        
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            half2 prod = __hmul2(a_half2[j], b_half2[j]);
            sum += __half2float(prod.x) + __half2float(prod.y);
        }
    }
    
    // Handle remainder
    for (int i = n_vec * 8 + tid; i < n; i += stride) {
        sum += __half2float(a[i]) * __half2float(b[i]);
    }
    
    sum = blockReduceSum(sum);
    
    if (threadIdx.x == 0) {
        atomicAdd(y, sum);
    }
}

// Host function to launch appropriate kernel
extern "C" void cuda_kernel(
    const void* a,
    const void* b,
    void* y,
    int n,
    int op_type
) {
    // Zero out the output before kernel launch
    cudaMemset(y, 0, sizeof(float));
    
    const int block_size = 256;
    const int grid_size = min((n + block_size - 1) / block_size, 1024);
    
    switch (op_type) {
        case 0: // f32 basic
            dot_product_f32<<<grid_size, block_size>>>(
                static_cast<const float*>(a),
                static_cast<const float*>(b),
                static_cast<float*>(y),
                n
            );
            break;
            
        case 1: // f32x4 vectorized
            dot_product_f32x4<<<grid_size, block_size>>>(
                static_cast<const float*>(a),
                static_cast<const float*>(b),
                static_cast<float*>(y),
                n
            );
            break;
            
        case 2: // f16 basic
            dot_product_f16<<<grid_size, block_size>>>(
                static_cast<const half*>(a),
                static_cast<const half*>(b),
                static_cast<float*>(y),
                n
            );
            break;
            
        case 3: // f16x2 vectorized
            dot_product_f16x2<<<grid_size, block_size>>>(
                static_cast<const half*>(a),
                static_cast<const half*>(b),
                static_cast<float*>(y),
                n
            );
            break;
            
        case 4: // f16x8 packed vectorized
            dot_product_f16x8<<<grid_size, block_size>>>(
                static_cast<const half*>(a),
                static_cast<const half*>(b),
                static_cast<float*>(y),
                n
            );
            break;
            
        default:
            // Default to basic f32
            dot_product_f32<<<grid_size, block_size>>>(
                static_cast<const float*>(a),
                static_cast<const float*>(b),
                static_cast<float*>(y),
                n
            );
            break;
    }
}
