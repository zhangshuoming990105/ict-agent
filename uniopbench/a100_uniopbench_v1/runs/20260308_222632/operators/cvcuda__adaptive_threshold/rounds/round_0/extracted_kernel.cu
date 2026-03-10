#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>

// Build Gaussian kernel weights
__global__ void build_gaussian_kernel_gpu(
    float* kernel,
    int K,
    float* sum_out)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= K * K) return;

    float sigma = 0.3f * ((K - 1) * 0.5f - 1.0f) + 0.8f;
    float cx = (K - 1) * 0.5f;
    
    int x = idx % K;
    int y = idx / K;
    
    float dx = x - cx;
    float dy = y - cx;
    
    float w = expf(-(dx * dx / (2.0f * sigma * sigma) + 
                     dy * dy / (2.0f * sigma * sigma)));
    
    kernel[idx] = w;
    
    // Use atomic add for sum
    atomicAdd(sum_out, w);
}

__global__ void normalize_kernel_gpu(float* kernel, int K, float sum) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= K * K) return;
    kernel[idx] /= sum;
}

__global__ void fill_mean_kernel_gpu(float* kernel, int K, float val) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= K * K) return;
    kernel[idx] = val;
}

// Compute local weighted sum using the pre-built kernel
__global__ void compute_local_mean(
    const uint8_t* src,
    const float* weight,
    float* local_mean,
    int N, int H, int W,
    int K)
{
    int n = blockIdx.z;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (n >= N || row >= H || col >= W) return;
    
    int pad = K / 2;
    float sum = 0.0f;
    
    for (int ky = 0; ky < K; ++ky) {
        for (int kx = 0; kx < K; ++kx) {
            int sy = row + ky - pad;
            int sx = col + kx - pad;
            
            // Replicate padding
            sy = max(0, min(H - 1, sy));
            sx = max(0, min(W - 1, sx));
            
            int src_idx = n * H * W + sy * W + sx;
            int k_idx = ky * K + kx;
            
            sum += (float)src[src_idx] * weight[k_idx];
        }
    }
    
    int out_idx = n * H * W + row * W + col;
    local_mean[out_idx] = sum;
}

// Apply adaptive threshold
__global__ void apply_threshold(
    const uint8_t* src,
    const float* local_mean,
    uint8_t* dst,
    int N, int H, int W,
    int threshold_type,
    float max_value,
    float c)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * H * W;
    
    if (idx >= total) return;
    
    int idelta;
    if (threshold_type == 0) {
        idelta = (int)ceilf(c);
    } else {
        idelta = (int)floorf(c);
    }
    
    int sval = (int)src[idx] + idelta;
    int res_i = (int)floorf(local_mean[idx] + 0.5f);
    res_i = max(0, min(255, res_i));
    
    bool cond;
    if (threshold_type == 0) {
        cond = sval > res_i;
    } else {
        cond = sval <= res_i;
    }
    
    int maxv = (int)roundf(max_value);
    dst[idx] = cond ? (uint8_t)maxv : (uint8_t)0;
}

extern "C" void cuda_kernel(
    uint8_t* src,
    float* temp_kernel,
    float* temp_sum,
    uint8_t* dst,
    int N, int H, int W,
    int adaptive_method,
    int threshold_type,
    int block_size,
    float max_value,
    float c)
{
    int K = block_size;
    int k_area = K * K;
    
    // Step 1: Build kernel weights
    if (adaptive_method == 0) {
        // Mean kernel
        float val = 1.0f / (float)(k_area);
        
        int threads = 256;
        int blocks = (k_area + threads - 1) / threads;
        
        fill_mean_kernel_gpu<<<blocks, threads>>>(temp_kernel, K, val);
        
    } else {
        // Gaussian kernel
        cudaMemset(temp_sum, 0, sizeof(float));
        
        int threads = 256;
        int blocks = (k_area + threads - 1) / threads;
        
        build_gaussian_kernel_gpu<<<blocks, threads>>>(
            temp_kernel, K, temp_sum);
        
        // Read sum back and normalize
        float h_sum;
        cudaMemcpy(&h_sum, temp_sum, sizeof(float), cudaMemcpyDeviceToHost);
        
        normalize_kernel_gpu<<<blocks, threads>>>(
            temp_kernel, K, h_sum);
    }
    
    // Step 2: Allocate and compute local mean
    float* local_mean;
    cudaMalloc(&local_mean, N * H * W * sizeof(float));
    
    dim3 threads_2d(16, 16);
    dim3 blocks_2d(
        (W + threads_2d.x - 1) / threads_2d.x,
        (H + threads_2d.y - 1) / threads_2d.y,
        N
    );
    
    compute_local_mean<<<blocks_2d, threads_2d>>>(
        src, temp_kernel, local_mean,
        N, H, W, K);
    
    // Step 3: Apply threshold
    int total_pixels = N * H * W;
    int threads_1d = 256;
    int blocks_1d = (total_pixels + threads_1d - 1) / threads_1d;
    
    apply_threshold<<<blocks_1d, threads_1d>>>(
        src, local_mean, dst,
        N, H, W, threshold_type, max_value, c);
    
    cudaFree(local_mean);
    cudaDeviceSynchronize();
}
