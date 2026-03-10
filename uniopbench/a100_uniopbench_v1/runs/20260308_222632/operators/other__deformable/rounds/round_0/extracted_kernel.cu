#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdio.h>

// Bilinear interpolation with boundary checks
__device__ __forceinline__ float bilinear_interpolate(
    const float* value_ptr,
    const int height,
    const int width,
    const int num_heads,
    const int channels,
    float h,
    float w,
    const int m,
    const int c
) {
    // Clamp to valid range
    if (h < 0 || h > height - 1 || w < 0 || w > width - 1) {
        return 0.0f;
    }

    // Get integer and fractional parts
    int h_low = floorf(h);
    int w_low = floorf(w);
    int h_high = h_low + 1;
    int w_high = w_low + 1;

    float lh = h - h_low;
    float lw = w - w_low;
    float hh = 1.0f - lh;
    float hw = 1.0f - lw;

    // Compute indices with boundary checks
    float v1 = 0.0f, v2 = 0.0f, v3 = 0.0f, v4 = 0.0f;

    if (h_low >= 0 && w_low >= 0) {
        int idx = ((h_low * width + w_low) * num_heads + m) * channels + c;
        v1 = value_ptr[idx];
    }
    if (h_low >= 0 && w_high <= width - 1) {
        int idx = ((h_low * width + w_high) * num_heads + m) * channels + c;
        v2 = value_ptr[idx];
    }
    if (h_high <= height - 1 && w_low >= 0) {
        int idx = ((h_high * width + w_low) * num_heads + m) * channels + c;
        v3 = value_ptr[idx];
    }
    if (h_high <= height - 1 && w_high <= width - 1) {
        int idx = ((h_high * width + w_high) * num_heads + m) * channels + c;
        v4 = value_ptr[idx];
    }

    // Bilinear interpolation
    float w1 = hh * hw;
    float w2 = hh * lw;
    float w3 = lh * hw;
    float w4 = lh * lw;

    return w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4;
}

// Main kernel for multi-scale deformable attention
__global__ void ms_deformable_attention_kernel(
    const float* value,                    // (N, S, M, D)
    const int* spatial_shapes,             // (L, 2)
    const int* level_start_index,          // (L,)
    const float* sampling_locations,       // (N, Lq, M, L, P, 2)
    const float* attention_weights,        // (N, Lq, M, L, P)
    float* output,                         // (N, Lq, M*D)
    const int batch_size,                  // N
    const int spatial_size,                // S (total spatial size)
    const int num_queries,                 // Lq
    const int num_heads,                   // M
    const int channels,                    // D
    const int num_levels,                  // L
    const int num_points                   // P
) {
    // Global thread index
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = batch_size * num_queries * num_heads * channels;

    if (index >= total_threads) return;

    // Decode thread index to (n, q, m, c)
    int c = index % channels;
    int m = (index / channels) % num_heads;
    int q = (index / (channels * num_heads)) % num_queries;
    int n = index / (channels * num_heads * num_queries);

    float sum = 0.0f;

    // Iterate over all levels and points
    for (int l = 0; l < num_levels; l++) {
        int level_start = level_start_index[l];
        int h = spatial_shapes[l * 2];
        int w = spatial_shapes[l * 2 + 1];

        // Pointer to value for this level
        // value layout: (N, S, M, D)
        const float* value_level_ptr = value + (n * spatial_size + level_start) * num_heads * channels;

        for (int p = 0; p < num_points; p++) {
            // Get sampling location (normalized 0-1)
            int loc_idx = ((((n * num_queries + q) * num_heads + m) * num_levels + l) * num_points + p) * 2;
            float loc_h = sampling_locations[loc_idx];
            float loc_w = sampling_locations[loc_idx + 1];

            // Convert to absolute coordinates
            float h_abs = loc_h * (h - 1);
            float w_abs = loc_w * (w - 1);

            // Get attention weight
            int weight_idx = (((n * num_queries + q) * num_heads + m) * num_levels + l) * num_points + p;
            float attn_weight = attention_weights[weight_idx];

            // Bilinear interpolation
            float value_interp = bilinear_interpolate(
                value_level_ptr,
                h, w,
                num_heads, channels,
                h_abs, w_abs,
                m, c
            );

            sum += value_interp * attn_weight;
        }
    }

    // Write output
    int out_idx = (n * num_queries + q) * (num_heads * channels) + m * channels + c;
    output[out_idx] = sum;
}

extern "C" {

void cuda_kernel(
    const float* value,
    const int* spatial_shapes,
    const int* level_start_index,
    const float* sampling_locations,
    const float* attention_weights,
    float* output
) {
    // Fixed parameters from get_data.py
    const int batch_size = 1;
    const int num_queries = 200;
    const int num_heads = 8;
    const int channels = 512;
    const int num_levels = 4;
    const int num_points = 4;
    
    // Calculate total spatial size: 32*32 + 16*16 + 8*8 + 4*4 = 1024 + 256 + 64 + 16 = 1360
    const int spatial_size = 1360;

    int total_threads = batch_size * num_queries * num_heads * channels;
    int threads_per_block = 256;
    int num_blocks = (total_threads + threads_per_block - 1) / threads_per_block;

    ms_deformable_attention_kernel<<<num_blocks, threads_per_block>>>(
        value,
        spatial_shapes,
        level_start_index,
        sampling_locations,
        attention_weights,
        output,
        batch_size,
        spatial_size,
        num_queries,
        num_heads,
        channels,
        num_levels,
        num_points
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA kernel launch error: %s\n", cudaGetErrorString(err));
    }
}

} // extern "C"
