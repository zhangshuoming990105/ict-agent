#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>
#include <stdint.h>

// Warp-level reduction for sum
__device__ __forceinline__ float warpReduceSum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Block-level reduction for sum
__device__ __forceinline__ float blockReduceSum(float val) {
    __shared__ float shared[32]; // One per warp
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;
    
    val = warpReduceSum(val);
    
    if (lane == 0) shared[wid] = val;
    __syncthreads();
    
    val = (threadIdx.x < blockDim.x / 32) ? shared[lane] : 0.0f;
    if (wid == 0) val = warpReduceSum(val);
    
    return val;
}

// Check if pointer is aligned for float4 access
__device__ __forceinline__ bool is_aligned_float4(const void* ptr) {
    return (reinterpret_cast<uintptr_t>(ptr) % 16) == 0;
}

// Optimized kernel using vectorized loads when possible and aligned
__global__ void instance_norm_forward_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    int batch,
    int channels,
    int spatial
) {
    const float eps = 1e-5f;
    
    // Each block processes one (batch, channel) instance
    int bc_idx = blockIdx.x;
    int b = bc_idx / channels;
    int c = bc_idx % channels;
    
    if (b >= batch || c >= channels) return;
    
    // Compute offset for this instance
    int offset = (b * channels + c) * spatial;
    const float* in_ptr = input + offset;
    float* out_ptr = output + offset;
    
    // Check if we can use vectorized loads (requires alignment and sufficient size)
    bool use_vectorized = (spatial >= 4) && is_aligned_float4(in_ptr) && is_aligned_float4(out_ptr);
    
    // Phase 1: Compute mean
    float sum = 0.0f;
    
    if (use_vectorized) {
        int vec_spatial = (spatial / 4) * 4;
        int vec_count = vec_spatial / 4;
        
        for (int i = threadIdx.x; i < vec_count; i += blockDim.x) {
            float4 data = reinterpret_cast<const float4*>(in_ptr)[i];
            sum += data.x + data.y + data.z + data.w;
        }
        
        // Handle remaining elements
        for (int i = vec_spatial + threadIdx.x; i < spatial; i += blockDim.x) {
            sum += in_ptr[i];
        }
    } else {
        // Scalar path
        for (int i = threadIdx.x; i < spatial; i += blockDim.x) {
            sum += in_ptr[i];
        }
    }
    
    sum = blockReduceSum(sum);
    
    __shared__ float s_mean;
    if (threadIdx.x == 0) {
        s_mean = sum / spatial;
    }
    __syncthreads();
    float mean = s_mean;
    
    // Phase 2: Compute variance
    float var_sum = 0.0f;
    
    if (use_vectorized) {
        int vec_spatial = (spatial / 4) * 4;
        int vec_count = vec_spatial / 4;
        
        for (int i = threadIdx.x; i < vec_count; i += blockDim.x) {
            float4 data = reinterpret_cast<const float4*>(in_ptr)[i];
            float diff_x = data.x - mean;
            float diff_y = data.y - mean;
            float diff_z = data.z - mean;
            float diff_w = data.w - mean;
            var_sum += diff_x * diff_x + diff_y * diff_y + diff_z * diff_z + diff_w * diff_w;
        }
        
        for (int i = vec_spatial + threadIdx.x; i < spatial; i += blockDim.x) {
            float diff = in_ptr[i] - mean;
            var_sum += diff * diff;
        }
    } else {
        for (int i = threadIdx.x; i < spatial; i += blockDim.x) {
            float diff = in_ptr[i] - mean;
            var_sum += diff * diff;
        }
    }
    
    var_sum = blockReduceSum(var_sum);
    
    __shared__ float s_invstd;
    if (threadIdx.x == 0) {
        float variance = var_sum / spatial;
        s_invstd = rsqrtf(variance + eps);
    }
    __syncthreads();
    float invstd = s_invstd;
    
    // Phase 3: Normalize and apply affine transformation
    float scale = gamma[c];
    float bias = beta[c];
    
    if (use_vectorized) {
        int vec_spatial = (spatial / 4) * 4;
        int vec_count = vec_spatial / 4;
        
        for (int i = threadIdx.x; i < vec_count; i += blockDim.x) {
            float4 data = reinterpret_cast<const float4*>(in_ptr)[i];
            float4 result;
            result.x = (data.x - mean) * invstd * scale + bias;
            result.y = (data.y - mean) * invstd * scale + bias;
            result.z = (data.z - mean) * invstd * scale + bias;
            result.w = (data.w - mean) * invstd * scale + bias;
            reinterpret_cast<float4*>(out_ptr)[i] = result;
        }
        
        for (int i = vec_spatial + threadIdx.x; i < spatial; i += blockDim.x) {
            float normalized = (in_ptr[i] - mean) * invstd;
            out_ptr[i] = normalized * scale + bias;
        }
    } else {
        for (int i = threadIdx.x; i < spatial; i += blockDim.x) {
            float normalized = (in_ptr[i] - mean) * invstd;
            out_ptr[i] = normalized * scale + bias;
        }
    }
}

extern "C" {

void cuda_kernel(
    const float* input,
    float* output,
    const float* gamma,
    const float* beta,
    int batch,
    int channels,
    int spatial
) {
    // Launch one block per (batch, channel) instance
    int total_instances = batch * channels;
    
    // Adjust thread count based on spatial size for better occupancy
    int threads;
    if (spatial <= 1024) {
        threads = 128;
    } else if (spatial <= 4096) {
        threads = 256;
    } else {
        threads = 512;
    }
    
    instance_norm_forward_kernel<<<total_instances, threads>>>(
        input, output, gamma, beta, batch, channels, spatial
    );
}

} // extern "C"
