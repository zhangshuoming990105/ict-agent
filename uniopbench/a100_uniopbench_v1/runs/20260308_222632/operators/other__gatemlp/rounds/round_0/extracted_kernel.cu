#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cmath>

// SiLU activation: x * sigmoid(x) = x / (1 + exp(-x))
__device__ __forceinline__ float silu(float x) {
    return x / (1.0f + expf(-x));
}

// Optimized GateMLP kernel for A100
// Computes: output = silu(X @ A) * (X @ B)
// X: (batch, K) fp16
// A, B: (K, N) fp16
// output: (batch, N) fp32
template<int BLOCK_SIZE = 256>
__global__ void gatemlp_kernel(
    const __half* __restrict__ X,
    const __half* __restrict__ A,
    const __half* __restrict__ B,
    float* __restrict__ output,
    int batch,
    int K,
    int N
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch * N;
    
    // Grid-stride loop
    for (int idx = tid; idx < total_elements; idx += blockDim.x * gridDim.x) {
        int b = idx / N;  // batch index
        int n = idx % N;  // output column index
        
        float sum_o1 = 0.0f;
        float sum_o2 = 0.0f;
        
        // Compute dot products: o1[b,n] = X[b,:] @ A[:,n], o2[b,n] = X[b,:] @ B[:,n]
        // Vectorized loads for better memory throughput
        int k = 0;
        
        // Process 4 elements at a time using half2
        for (; k + 3 < K; k += 4) {
            // Load X values
            float x0 = __half2float(X[b * K + k]);
            float x1 = __half2float(X[b * K + k + 1]);
            float x2 = __half2float(X[b * K + k + 2]);
            float x3 = __half2float(X[b * K + k + 3]);
            
            // Load A values (column n)
            float a0 = __half2float(A[k * N + n]);
            float a1 = __half2float(A[(k + 1) * N + n]);
            float a2 = __half2float(A[(k + 2) * N + n]);
            float a3 = __half2float(A[(k + 3) * N + n]);
            
            // Load B values (column n)
            float b0 = __half2float(B[k * N + n]);
            float b1 = __half2float(B[(k + 1) * N + n]);
            float b2 = __half2float(B[(k + 2) * N + n]);
            float b3 = __half2float(B[(k + 3) * N + n]);
            
            // Accumulate
            sum_o1 += x0 * a0 + x1 * a1 + x2 * a2 + x3 * a3;
            sum_o2 += x0 * b0 + x1 * b1 + x2 * b2 + x3 * b3;
        }
        
        // Handle remaining elements
        for (; k < K; k++) {
            float x_val = __half2float(X[b * K + k]);
            float a_val = __half2float(A[k * N + n]);
            float b_val = __half2float(B[k * N + n]);
            sum_o1 += x_val * a_val;
            sum_o2 += x_val * b_val;
        }
        
        // Apply gating: silu(o1) * o2
        output[idx] = silu(sum_o1) * sum_o2;
    }
}

extern "C" {

void cuda_kernel(
    const void* X,
    const void* A,
    const void* B,
    void* output,
    int batch,
    int K,
    int N
) {
    const int BLOCK_SIZE = 256;
    int total_elements = batch * N;
    int num_blocks = (total_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    // Limit grid size for better occupancy
    num_blocks = min(num_blocks, 1024);
    
    gatemlp_kernel<BLOCK_SIZE><<<num_blocks, BLOCK_SIZE>>>(
        reinterpret_cast<const __half*>(X),
        reinterpret_cast<const __half*>(A),
        reinterpret_cast<const __half*>(B),
        reinterpret_cast<float*>(output),
        batch,
        K,
        N
    );
}

} // extern "C"
