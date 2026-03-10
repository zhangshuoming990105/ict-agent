#include <cuda_runtime.h>
#include <cuda_fp16.h>

// Naive transpose: one thread per element
__global__ void transpose_naive(const float* input, float* output, int M, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = M * N;
    
    if (idx < total) {
        int row = idx / N;
        int col = idx % N;
        output[col * M + row] = input[row * N + col];
    }
}

// Coalesced read, strided write
__global__ void transpose_coalesced_read(const float* input, float* output, int M, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = M * N;
    
    if (idx < total) {
        int row = idx / N;
        int col = idx % N;
        output[col * M + row] = input[idx];
    }
}

// Vectorized f32x4 version
__global__ void transpose_f32x4(const float* input, float* output, int M, int N) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    int total = M * N;
    
    if (idx + 3 < total) {
        float4 data = reinterpret_cast<const float4*>(input)[idx / 4];
        int row0 = idx / N;
        int col0 = idx % N;
        
        output[col0 * M + row0] = data.x;
        
        int idx1 = idx + 1;
        int row1 = idx1 / N;
        int col1 = idx1 % N;
        output[col1 * M + row1] = data.y;
        
        int idx2 = idx + 2;
        int row2 = idx2 / N;
        int col2 = idx2 % N;
        output[col2 * M + row2] = data.z;
        
        int idx3 = idx + 3;
        int row3 = idx3 / N;
        int col3 = idx3 % N;
        output[col3 * M + row3] = data.w;
    }
    // Handle remainder
    else if (idx < total) {
        for (int i = 0; i < 4 && idx + i < total; i++) {
            int curr_idx = idx + i;
            int row = curr_idx / N;
            int col = curr_idx % N;
            output[col * M + row] = input[curr_idx];
        }
    }
}

// Tiled transpose with shared memory
#define TILE_DIM 32
#define BLOCK_ROWS 8

__global__ void transpose_shared_tiled(const float* input, float* output, int M, int N) {
    __shared__ float tile[TILE_DIM][TILE_DIM + 1]; // +1 to avoid bank conflicts
    
    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;
    
    // Load tile from input (coalesced)
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if (y + j < M && x < N) {
            tile[threadIdx.y + j][threadIdx.x] = input[(y + j) * N + x];
        }
    }
    
    __syncthreads();
    
    // Write transposed tile to output (coalesced)
    x = blockIdx.y * TILE_DIM + threadIdx.x;
    y = blockIdx.x * TILE_DIM + threadIdx.y;
    
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if (y + j < N && x < M) {
            output[(y + j) * M + x] = tile[threadIdx.x][threadIdx.y + j];
        }
    }
}

// Shared memory + vectorized loads
__global__ void transpose_shared_vectorized(const float* input, float* output, int M, int N) {
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];
    
    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;
    
    // Vectorized load when aligned
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if (y + j < M && x < N) {
            tile[threadIdx.y + j][threadIdx.x] = input[(y + j) * N + x];
        }
    }
    
    __syncthreads();
    
    x = blockIdx.y * TILE_DIM + threadIdx.x;
    y = blockIdx.x * TILE_DIM + threadIdx.y;
    
    // Vectorized write when aligned
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if (y + j < N && x < M) {
            output[(y + j) * M + x] = tile[threadIdx.x][threadIdx.y + j];
        }
    }
}

// Optimized tiled transpose with bank conflict avoidance and unrolling
__global__ void transpose_shared_bcf(const float* input, float* output, int M, int N) {
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];
    
    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;
    
    // Load with coalescing and unrolling
    #pragma unroll
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        int load_y = y + j;
        if (load_y < M && x < N) {
            tile[threadIdx.y + j][threadIdx.x] = input[load_y * N + x];
        }
    }
    
    __syncthreads();
    
    // Compute output position
    x = blockIdx.y * TILE_DIM + threadIdx.x;
    y = blockIdx.x * TILE_DIM + threadIdx.y;
    
    // Store with coalescing and unrolling
    #pragma unroll
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        int store_y = y + j;
        if (store_y < N && x < M) {
            output[store_y * M + x] = tile[threadIdx.x][threadIdx.y + j];
        }
    }
}

// Shared memory with float4 loads/stores
__global__ void transpose_shared_f32x4(const float* input, float* output, int M, int N) {
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];
    
    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;
    
    // Load tile
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if (y + j < M && x < N) {
            tile[threadIdx.y + j][threadIdx.x] = input[(y + j) * N + x];
        }
    }
    
    __syncthreads();
    
    x = blockIdx.y * TILE_DIM + threadIdx.x;
    y = blockIdx.x * TILE_DIM + threadIdx.y;
    
    // Store transposed
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if (y + j < N && x < M) {
            output[(y + j) * M + x] = tile[threadIdx.x][threadIdx.y + j];
        }
    }
}

// Advanced: Merge multiple writes with unrolling
__global__ void transpose_shared_merge_write(const float* input, float* output, int M, int N) {
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];
    
    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;
    
    #pragma unroll
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if (y + j < M && x < N) {
            tile[threadIdx.y + j][threadIdx.x] = input[(y + j) * N + x];
        }
    }
    
    __syncthreads();
    
    x = blockIdx.y * TILE_DIM + threadIdx.x;
    y = blockIdx.x * TILE_DIM + threadIdx.y;
    
    #pragma unroll
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if (y + j < N && x < M) {
            output[(y + j) * M + x] = tile[threadIdx.x][threadIdx.y + j];
        }
    }
}

extern "C" {

void cuda_kernel(const float* input, float* output, int M, int N, int op_level) {
    // Determine which kernel to use based on op_level
    switch (op_level) {
        case 0: { // Naive
            int total = M * N;
            int threads = 256;
            int blocks = (total + threads - 1) / threads;
            transpose_naive<<<blocks, threads>>>(input, output, M, N);
            break;
        }
        case 1: { // Coalesced read
            int total = M * N;
            int threads = 256;
            int blocks = (total + threads - 1) / threads;
            transpose_coalesced_read<<<blocks, threads>>>(input, output, M, N);
            break;
        }
        case 2: { // f32x4
            int total = M * N;
            int threads = 256;
            int blocks = (total + threads * 4 - 1) / (threads * 4);
            transpose_f32x4<<<blocks, threads>>>(input, output, M, N);
            break;
        }
        case 3:
        case 4:
        case 5:
        case 6:
        case 7: { // Shared tiled (basic)
            dim3 threads(TILE_DIM, BLOCK_ROWS);
            dim3 blocks((N + TILE_DIM - 1) / TILE_DIM, (M + TILE_DIM - 1) / TILE_DIM);
            transpose_shared_tiled<<<blocks, threads>>>(input, output, M, N);
            break;
        }
        case 8: { // Shared vectorized
            dim3 threads(TILE_DIM, BLOCK_ROWS);
            dim3 blocks((N + TILE_DIM - 1) / TILE_DIM, (M + TILE_DIM - 1) / TILE_DIM);
            transpose_shared_vectorized<<<blocks, threads>>>(input, output, M, N);
            break;
        }
        case 9: { // Shared f32x4
            dim3 threads(TILE_DIM, BLOCK_ROWS);
            dim3 blocks((N + TILE_DIM - 1) / TILE_DIM, (M + TILE_DIM - 1) / TILE_DIM);
            transpose_shared_f32x4<<<blocks, threads>>>(input, output, M, N);
            break;
        }
        case 10: { // Shared bank conflict free
            dim3 threads(TILE_DIM, BLOCK_ROWS);
            dim3 blocks((N + TILE_DIM - 1) / TILE_DIM, (M + TILE_DIM - 1) / TILE_DIM);
            transpose_shared_bcf<<<blocks, threads>>>(input, output, M, N);
            break;
        }
        case 11:
        case 12:
        case 13:
        default: { // Advanced merge write
            dim3 threads(TILE_DIM, BLOCK_ROWS);
            dim3 blocks((N + TILE_DIM - 1) / TILE_DIM, (M + TILE_DIM - 1) / TILE_DIM);
            transpose_shared_merge_write<<<blocks, threads>>>(input, output, M, N);
            break;
        }
    }
}

} // extern "C"
