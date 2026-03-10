#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <stdio.h>

using namespace nvcuda;

// ============================================================================
// Kernel 0: Naive SGEMM (float32)
// ============================================================================
__global__ void sgemm_naive(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// ============================================================================
// Kernel 1: SlicedK SGEMM (float32) - Grid-stride loop over K
// ============================================================================
__global__ void sgemm_sliced_k(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        float sum = 0.0f;
        // Grid-stride loop over K for better memory coalescing
        for (int k = 0; k < K; k += 8) {
            #pragma unroll
            for (int kk = 0; kk < 8 && (k + kk) < K; ++kk) {
                sum += A[row * K + k + kk] * B[(k + kk) * N + col];
            }
        }
        C[row * N + col] = sum;
    }
}

// ============================================================================
// Kernel 2: Tiled 8x8 Block Column Format SGEMM (float32)
// ============================================================================
#define TILE_SIZE 8

__global__ void sgemm_tiled_8x8(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K)
{
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;
    
    float sum = 0.0f;
    
    int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;
    
    for (int t = 0; t < numTiles; ++t) {
        // Load tiles into shared memory
        int a_col = t * TILE_SIZE + tx;
        int b_row = t * TILE_SIZE + ty;
        
        As[ty][tx] = (row < M && a_col < K) ? A[row * K + a_col] : 0.0f;
        Bs[ty][tx] = (b_row < K && col < N) ? B[b_row * N + col] : 0.0f;
        
        __syncthreads();
        
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += As[ty][k] * Bs[k][tx];
        }
        
        __syncthreads();
    }
    
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// ============================================================================
// Kernel 3: Optimized Tiled SGEMM with 16x16 blocks (float32)
// ============================================================================
#define TILE_SIZE_16 16

__global__ void sgemm_wmma_tf32(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K)
{
    __shared__ float As[TILE_SIZE_16][TILE_SIZE_16];
    __shared__ float Bs[TILE_SIZE_16][TILE_SIZE_16];
    
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    int row = by * TILE_SIZE_16 + ty;
    int col = bx * TILE_SIZE_16 + tx;
    
    float sum = 0.0f;
    
    int numTiles = (K + TILE_SIZE_16 - 1) / TILE_SIZE_16;
    
    for (int t = 0; t < numTiles; ++t) {
        // Load tiles into shared memory
        int a_col = t * TILE_SIZE_16 + tx;
        int b_row = t * TILE_SIZE_16 + ty;
        
        As[ty][tx] = (row < M && a_col < K) ? A[row * K + a_col] : 0.0f;
        Bs[ty][tx] = (b_row < K && col < N) ? B[b_row * N + col] : 0.0f;
        
        __syncthreads();
        
        #pragma unroll
        for (int k = 0; k < TILE_SIZE_16; ++k) {
            sum += As[ty][k] * Bs[k][tx];
        }
        
        __syncthreads();
    }
    
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// ============================================================================
// Kernel 4: Naive HGEMM (float16)
// ============================================================================
__global__ void hgemm_naive(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C,
    int M, int N, int K)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += __half2float(A[row * K + k]) * __half2float(B[k * N + col]);
        }
        C[row * N + col] = __float2half(sum);
    }
}

// ============================================================================
// Kernel 5: Tiled 8x8 Block Column Format HGEMM (float16)
// ============================================================================
__global__ void hgemm_tiled_8x8(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C,
    int M, int N, int K)
{
    __shared__ half As[TILE_SIZE][TILE_SIZE];
    __shared__ half Bs[TILE_SIZE][TILE_SIZE];
    
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;
    
    float sum = 0.0f;
    
    int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;
    
    for (int t = 0; t < numTiles; ++t) {
        int a_col = t * TILE_SIZE + tx;
        int b_row = t * TILE_SIZE + ty;
        
        As[ty][tx] = (row < M && a_col < K) ? A[row * K + a_col] : __float2half(0.0f);
        Bs[ty][tx] = (b_row < K && col < N) ? B[b_row * N + col] : __float2half(0.0f);
        
        __syncthreads();
        
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += __half2float(As[ty][k]) * __half2float(Bs[k][tx]);
        }
        
        __syncthreads();
    }
    
    if (row < M && col < N) {
        C[row * N + col] = __float2half(sum);
    }
}

// ============================================================================
// Kernel 6: MMA Tensor Core Stage 2 (float16) with half accumulator
// ============================================================================
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

__global__ void hgemm_mma_stage2(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C,
    int M, int N, int K)
{
    // Each block handles one output tile
    int warp_row = blockIdx.y * WMMA_M;
    int warp_col = blockIdx.x * WMMA_N;
    
    if (warp_row >= M || warp_col >= N) return;
    
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
    
    wmma::fill_fragment(c_frag, 0.0f);
    
    int numTiles = (K + WMMA_K - 1) / WMMA_K;
    
    for (int t = 0; t < numTiles; ++t) {
        int k_offset = t * WMMA_K;
        
        if (warp_row + WMMA_M <= M && k_offset + WMMA_K <= K) {
            wmma::load_matrix_sync(a_frag, A + warp_row * K + k_offset, K);
        } else {
            wmma::fill_fragment(a_frag, __float2half(0.0f));
        }
        
        if (k_offset + WMMA_K <= K && warp_col + WMMA_N <= N) {
            wmma::load_matrix_sync(b_frag, B + k_offset * N + warp_col, N);
        } else {
            wmma::fill_fragment(b_frag, __float2half(0.0f));
        }
        
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }
    
    // Convert float accumulator to half and store
    if (warp_row + WMMA_M <= M && warp_col + WMMA_N <= N) {
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> c_frag_half;
        
        // Manual conversion from float to half fragment
        for (int i = 0; i < c_frag.num_elements; i++) {
            c_frag_half.x[i] = __float2half(c_frag.x[i]);
        }
        
        wmma::store_matrix_sync(C + warp_row * N + warp_col, c_frag_half, N, wmma::mem_row_major);
    }
}

// ============================================================================
// Kernel 7: MMA Tensor Core Stage 1 (float16) - with float accumulator
// ============================================================================
__global__ void hgemm_mma_stage1(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C,
    int M, int N, int K)
{
    // Each block handles one output tile
    int warp_row = blockIdx.y * WMMA_M;
    int warp_col = blockIdx.x * WMMA_N;
    
    if (warp_row >= M || warp_col >= N) return;
    
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
    
    wmma::fill_fragment(c_frag, 0.0f);
    
    int numTiles = (K + WMMA_K - 1) / WMMA_K;
    
    for (int t = 0; t < numTiles; ++t) {
        int k_offset = t * WMMA_K;
        
        if (warp_row + WMMA_M <= M && k_offset + WMMA_K <= K) {
            wmma::load_matrix_sync(a_frag, A + warp_row * K + k_offset, K);
        } else {
            wmma::fill_fragment(a_frag, __float2half(0.0f));
        }
        
        if (k_offset + WMMA_K <= K && warp_col + WMMA_N <= N) {
            wmma::load_matrix_sync(b_frag, B + k_offset * N + warp_col, N);
        } else {
            wmma::fill_fragment(b_frag, __float2half(0.0f));
        }
        
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }
    
    // Convert float accumulator to half and store
    if (warp_row + WMMA_M <= M && warp_col + WMMA_N <= N) {
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> c_frag_half;
        
        // Manual conversion from float to half fragment
        for (int i = 0; i < c_frag.num_elements; i++) {
            c_frag_half.x[i] = __float2half(c_frag.x[i]);
        }
        
        wmma::store_matrix_sync(C + warp_row * N + warp_col, c_frag_half, N, wmma::mem_row_major);
    }
}

// ============================================================================
// Host Interface - Dispatch to correct kernel based on op_type
// ============================================================================
extern "C" void cuda_kernel(
    void* a_ptr,
    void* b_ptr,
    void* c_ptr,
    int M,
    int N,
    int K,
    int op_type)
{
    // Choose kernel based on op_type
    switch (op_type) {
        case 0: { // Naive SGEMM
            dim3 block(16, 16);
            dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
            sgemm_naive<<<grid, block>>>(
                (const float*)a_ptr,
                (const float*)b_ptr,
                (float*)c_ptr,
                M, N, K);
            break;
        }
        case 1: { // SlicedK SGEMM
            dim3 block(16, 16);
            dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
            sgemm_sliced_k<<<grid, block>>>(
                (const float*)a_ptr,
                (const float*)b_ptr,
                (float*)c_ptr,
                M, N, K);
            break;
        }
        case 2: { // Tiled 8x8 SGEMM
            dim3 block(TILE_SIZE, TILE_SIZE);
            dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);
            sgemm_tiled_8x8<<<grid, block>>>(
                (const float*)a_ptr,
                (const float*)b_ptr,
                (float*)c_ptr,
                M, N, K);
            break;
        }
        case 3: { // Optimized Tiled 16x16 FP32
            dim3 block(TILE_SIZE_16, TILE_SIZE_16);
            dim3 grid((N + TILE_SIZE_16 - 1) / TILE_SIZE_16, (M + TILE_SIZE_16 - 1) / TILE_SIZE_16);
            sgemm_wmma_tf32<<<grid, block>>>(
                (const float*)a_ptr,
                (const float*)b_ptr,
                (float*)c_ptr,
                M, N, K);
            break;
        }
        case 4: { // Naive HGEMM
            dim3 block(16, 16);
            dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
            hgemm_naive<<<grid, block>>>(
                (const half*)a_ptr,
                (const half*)b_ptr,
                (half*)c_ptr,
                M, N, K);
            break;
        }
        case 5: { // Tiled 8x8 HGEMM
            dim3 block(TILE_SIZE, TILE_SIZE);
            dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);
            hgemm_tiled_8x8<<<grid, block>>>(
                (const half*)a_ptr,
                (const half*)b_ptr,
                (half*)c_ptr,
                M, N, K);
            break;
        }
        case 6: { // MMA Stage 2 HGEMM - one warp per block
            dim3 block(32);
            dim3 grid((N + WMMA_N - 1) / WMMA_N, (M + WMMA_M - 1) / WMMA_M);
            hgemm_mma_stage2<<<grid, block>>>(
                (const half*)a_ptr,
                (const half*)b_ptr,
                (half*)c_ptr,
                M, N, K);
            break;
        }
        case 7: { // MMA Stage 1 HGEMM - one warp per block
            dim3 block(32);
            dim3 grid((N + WMMA_N - 1) / WMMA_N, (M + WMMA_M - 1) / WMMA_M);
            hgemm_mma_stage1<<<grid, block>>>(
                (const half*)a_ptr,
                (const half*)b_ptr,
                (half*)c_ptr,
                M, N, K);
            break;
        }
        default:
            printf("Error: Unknown op_type %d\n", op_type);
            break;
    }
}
