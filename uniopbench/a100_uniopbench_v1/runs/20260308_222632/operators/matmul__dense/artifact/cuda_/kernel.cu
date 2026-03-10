#include <cuda_fp16.h>
#include <cuda_runtime.h>

// Optimized tile configuration for A100
constexpr int TILE_M = 128;
constexpr int TILE_N = 128;
constexpr int TILE_K = 32;
constexpr int BLOCK_SIZE = 256;

// Thread-level tiling
constexpr int THREAD_TILE_M = 8;
constexpr int THREAD_TILE_N = 8;

__global__ void matmul_bias_kernel(
    const half* __restrict__ A,
    const half* __restrict__ B,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int M, int K, int N
) {
    // Shared memory for tiling
    __shared__ half smem_A[TILE_M * TILE_K];
    __shared__ half smem_B[TILE_K * TILE_N];
    
    // Thread and block indices
    const int tid = threadIdx.x;
    const int block_row = blockIdx.y;
    const int block_col = blockIdx.x;
    
    const int row_base = block_row * TILE_M;
    const int col_base = block_col * TILE_N;
    
    // Thread-level output tile mapping
    // 256 threads handle 128x128 output = 16384 elements
    // Each thread handles 8x8 = 64 elements
    const int threads_per_row = TILE_N / THREAD_TILE_N;  // 128 / 8 = 16
    const int thread_row = (tid / threads_per_row) * THREAD_TILE_M;
    const int thread_col = (tid % threads_per_row) * THREAD_TILE_N;
    
    // Accumulator registers
    float acc[THREAD_TILE_M][THREAD_TILE_N];
    #pragma unroll
    for (int i = 0; i < THREAD_TILE_M; ++i) {
        #pragma unroll
        for (int j = 0; j < THREAD_TILE_N; ++j) {
            acc[i][j] = 0.0f;
        }
    }
    
    // Main K loop
    const int num_k_tiles = (K + TILE_K - 1) / TILE_K;
    
    for (int k_tile = 0; k_tile < num_k_tiles; ++k_tile) {
        const int k_base = k_tile * TILE_K;
        
        // Cooperative loading of A tile (TILE_M x TILE_K)
        const int A_tile_elements = TILE_M * TILE_K;
        const int loads_per_thread_A = (A_tile_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;
        
        #pragma unroll 4
        for (int l = 0; l < loads_per_thread_A; ++l) {
            int idx = tid + l * BLOCK_SIZE;
            if (idx < A_tile_elements) {
                int local_row = idx / TILE_K;
                int local_col = idx % TILE_K;
                int global_row = row_base + local_row;
                int global_col = k_base + local_col;
                
                half val = __float2half(0.0f);
                if (global_row < M && global_col < K) {
                    val = A[global_row * K + global_col];
                }
                smem_A[local_row * TILE_K + local_col] = val;
            }
        }
        
        // Cooperative loading of B tile (TILE_K x TILE_N)
        const int B_tile_elements = TILE_K * TILE_N;
        const int loads_per_thread_B = (B_tile_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;
        
        #pragma unroll 4
        for (int l = 0; l < loads_per_thread_B; ++l) {
            int idx = tid + l * BLOCK_SIZE;
            if (idx < B_tile_elements) {
                int local_row = idx / TILE_N;
                int local_col = idx % TILE_N;
                int global_row = k_base + local_row;
                int global_col = col_base + local_col;
                
                half val = __float2half(0.0f);
                if (global_row < K && global_col < N) {
                    val = B[global_row * N + global_col];
                }
                smem_B[local_row * TILE_N + local_col] = val;
            }
        }
        
        __syncthreads();
        
        // Compute thread-local matmul
        #pragma unroll
        for (int k = 0; k < TILE_K; ++k) {
            // Load A values for this thread's rows
            half a_vals[THREAD_TILE_M];
            #pragma unroll
            for (int i = 0; i < THREAD_TILE_M; ++i) {
                int row = thread_row + i;
                a_vals[i] = (row < TILE_M) ? smem_A[row * TILE_K + k] : __float2half(0.0f);
            }
            
            // Load B values for this thread's columns
            half b_vals[THREAD_TILE_N];
            #pragma unroll
            for (int j = 0; j < THREAD_TILE_N; ++j) {
                int col = thread_col + j;
                b_vals[j] = (col < TILE_N) ? smem_B[k * TILE_N + col] : __float2half(0.0f);
            }
            
            // Outer product update
            #pragma unroll
            for (int i = 0; i < THREAD_TILE_M; ++i) {
                float a_f = __half2float(a_vals[i]);
                #pragma unroll
                for (int j = 0; j < THREAD_TILE_N; ++j) {
                    acc[i][j] += a_f * __half2float(b_vals[j]);
                }
            }
        }
        
        __syncthreads();
    }
    
    // Write results to global memory with bias
    #pragma unroll
    for (int i = 0; i < THREAD_TILE_M; ++i) {
        int global_row = row_base + thread_row + i;
        if (global_row >= M) continue;
        
        #pragma unroll
        for (int j = 0; j < THREAD_TILE_N; ++j) {
            int global_col = col_base + thread_col + j;
            if (global_col < N) {
                output[global_row * N + global_col] = acc[i][j] + bias[global_col];
            }
        }
    }
}

extern "C" void cuda_kernel(
    const half* A,
    const half* B,
    const float* bias,
    float* output,
    int M, int K, int N
) {
    dim3 block(BLOCK_SIZE);
    dim3 grid(
        (N + TILE_N - 1) / TILE_N,
        (M + TILE_M - 1) / TILE_M
    );
    
    matmul_bias_kernel<<<grid, block>>>(A, B, bias, output, M, K, N);
}
