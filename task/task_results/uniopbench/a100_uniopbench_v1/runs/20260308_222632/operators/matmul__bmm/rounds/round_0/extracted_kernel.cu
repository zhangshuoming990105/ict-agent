#include <cuda_fp16.h>
#include <cuda_runtime.h>

// Optimized tile sizes for A100
#define BM 128
#define BN 128
#define BK 32
#define TM 8
#define TN 8

// Optimized BMM kernel using better memory access patterns
__global__ void bmm_kernel(
    const half* __restrict__ A,  // [batch, m, k]
    const half* __restrict__ B,  // [batch, k, n]
    float* __restrict__ C,       // [batch, m, n]
    int batch,
    int m,
    int k,
    int n
) {
    // Batch index
    int b = blockIdx.z;
    if (b >= batch) return;
    
    // Block tile position
    int bx = blockIdx.x;
    int by = blockIdx.y;
    
    // Thread indices
    int tx = threadIdx.x % 16;  // column within warp tile
    int ty = threadIdx.x / 16;  // row within warp tile
    
    // Thread's output position
    int warp_row = ty * TM;
    int warp_col = tx * TN;
    
    // Shared memory for tiles - padded to avoid bank conflicts
    __shared__ half As[BM][BK + 8];
    __shared__ half Bs[BK][BN + 8];
    
    // Registers for accumulation
    float acc[TM][TN];
    #pragma unroll
    for (int i = 0; i < TM; i++) {
        #pragma unroll
        for (int j = 0; j < TN; j++) {
            acc[i][j] = 0.0f;
        }
    }
    
    // Base pointers for this batch
    const half* A_batch = A + b * m * k;
    const half* B_batch = B + b * k * n;
    float* C_batch = C + b * m * n;
    
    // Global output position
    int c_row_start = by * BM;
    int c_col_start = bx * BN;
    
    // Thread's work assignment for loading
    int tid = threadIdx.x;
    int num_threads = blockDim.x;
    
    // Loop over K dimension
    for (int kb = 0; kb < k; kb += BK) {
        // Cooperatively load A tile [BM x BK]
        int a_load_count = (BM * BK + num_threads - 1) / num_threads;
        #pragma unroll 4
        for (int i = 0; i < a_load_count; i++) {
            int idx = tid + i * num_threads;
            if (idx < BM * BK) {
                int row = idx / BK;
                int col = idx % BK;
                int global_row = c_row_start + row;
                int global_col = kb + col;
                
                if (global_row < m && global_col < k) {
                    As[row][col] = A_batch[global_row * k + global_col];
                } else {
                    As[row][col] = __float2half(0.0f);
                }
            }
        }
        
        // Cooperatively load B tile [BK x BN]
        int b_load_count = (BK * BN + num_threads - 1) / num_threads;
        #pragma unroll 4
        for (int i = 0; i < b_load_count; i++) {
            int idx = tid + i * num_threads;
            if (idx < BK * BN) {
                int row = idx / BN;
                int col = idx % BN;
                int global_row = kb + row;
                int global_col = c_col_start + col;
                
                if (global_row < k && global_col < n) {
                    Bs[row][col] = B_batch[global_row * n + global_col];
                } else {
                    Bs[row][col] = __float2half(0.0f);
                }
            }
        }
        
        __syncthreads();
        
        // Compute: each thread computes TM x TN outputs
        #pragma unroll
        for (int kk = 0; kk < BK; kk++) {
            // Load A values
            half a_vals[TM];
            #pragma unroll
            for (int i = 0; i < TM; i++) {
                if (warp_row + i < BM) {
                    a_vals[i] = As[warp_row + i][kk];
                } else {
                    a_vals[i] = __float2half(0.0f);
                }
            }
            
            // Load B values
            half b_vals[TN];
            #pragma unroll
            for (int j = 0; j < TN; j++) {
                if (warp_col + j < BN) {
                    b_vals[j] = Bs[kk][warp_col + j];
                } else {
                    b_vals[j] = __float2half(0.0f);
                }
            }
            
            // Outer product
            #pragma unroll
            for (int i = 0; i < TM; i++) {
                #pragma unroll
                for (int j = 0; j < TN; j++) {
                    acc[i][j] += __half2float(a_vals[i]) * __half2float(b_vals[j]);
                }
            }
        }
        
        __syncthreads();
    }
    
    // Write results
    #pragma unroll
    for (int i = 0; i < TM; i++) {
        #pragma unroll
        for (int j = 0; j < TN; j++) {
            int row = c_row_start + warp_row + i;
            int col = c_col_start + warp_col + j;
            
            if (row < m && col < n) {
                C_batch[row * n + col] = acc[i][j];
            }
        }
    }
}

extern "C" {
    void cuda_kernel(
        const half* A,
        const half* B, 
        float* C,
        int batch,
        int m,
        int k,
        int n
    ) {
        // Use 256 threads per block (16x16 thread layout for TM=8, TN=8)
        const int THREADS = 256;
        
        dim3 block(THREADS);
        dim3 grid(
            (n + BN - 1) / BN,
            (m + BM - 1) / BM,
            batch
        );
        
        bmm_kernel<<<grid, block>>>(A, B, C, batch, m, k, n);
    }
}
