#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>

// GELU approximation using tanh: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
// Constants for tanh approximation
#define GELU_SCALING_FACTOR 0.7978845608028654f  // sqrt(2/pi)
#define GELU_COEFF 0.044715f

// =============== Float32 Kernels ===============

// OP_TYPE = 0: Float32 Scalar
__global__ void gelu_f32_scalar_kernel(const float* input, float* output, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float x = input[idx];
        float x_cubed = x * x * x;
        float inner = GELU_SCALING_FACTOR * (x + GELU_COEFF * x_cubed);
        float tanh_val = tanhf(inner);
        output[idx] = 0.5f * x * (1.0f + tanh_val);
    }
}

// OP_TYPE = 1: Float32 Vectorized (float4)
__global__ void gelu_f32_vec4_kernel(const float* input, float* output, int N) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    
    if (idx + 3 < N) {
        // Vectorized load
        float4 x = *reinterpret_cast<const float4*>(&input[idx]);
        float4 result;
        
        // Process each element
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            float val = reinterpret_cast<float*>(&x)[i];
            float x_cubed = val * val * val;
            float inner = GELU_SCALING_FACTOR * (val + GELU_COEFF * x_cubed);
            float tanh_val = tanhf(inner);
            reinterpret_cast<float*>(&result)[i] = 0.5f * val * (1.0f + tanh_val);
        }
        
        // Vectorized store
        *reinterpret_cast<float4*>(&output[idx]) = result;
    } else if (idx < N) {
        // Handle remaining elements
        for (int i = idx; i < N; i++) {
            float x = input[i];
            float x_cubed = x * x * x;
            float inner = GELU_SCALING_FACTOR * (x + GELU_COEFF * x_cubed);
            float tanh_val = tanhf(inner);
            output[i] = 0.5f * x * (1.0f + tanh_val);
        }
    }
}

// =============== Float16 Helper Functions ===============

__device__ __forceinline__ half gelu_f16_scalar(half x) {
    float x_f = __half2float(x);
    float x_cubed = x_f * x_f * x_f;
    float inner = GELU_SCALING_FACTOR * (x_f + GELU_COEFF * x_cubed);
    float tanh_val = tanhf(inner);
    float result = 0.5f * x_f * (1.0f + tanh_val);
    return __float2half(result);
}

__device__ __forceinline__ half2 gelu_f16_half2(half2 x) {
    float2 x_f = __half22float2(x);
    
    float x0_cubed = x_f.x * x_f.x * x_f.x;
    float inner0 = GELU_SCALING_FACTOR * (x_f.x + GELU_COEFF * x0_cubed);
    float tanh_val0 = tanhf(inner0);
    float result0 = 0.5f * x_f.x * (1.0f + tanh_val0);
    
    float x1_cubed = x_f.y * x_f.y * x_f.y;
    float inner1 = GELU_SCALING_FACTOR * (x_f.y + GELU_COEFF * x1_cubed);
    float tanh_val1 = tanhf(inner1);
    float result1 = 0.5f * x_f.y * (1.0f + tanh_val1);
    
    return __floats2half2_rn(result0, result1);
}

// =============== Float16 Kernels ===============

// OP_TYPE = 2: Float16 Scalar
__global__ void gelu_f16_scalar_kernel(const half* input, half* output, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        output[idx] = gelu_f16_scalar(input[idx]);
    }
}

// OP_TYPE = 3: Float16 Vectorized (half2)
__global__ void gelu_f16_vec2_kernel(const half* input, half* output, int N) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
    
    if (idx + 1 < N) {
        half2 x = *reinterpret_cast<const half2*>(&input[idx]);
        half2 result = gelu_f16_half2(x);
        *reinterpret_cast<half2*>(&output[idx]) = result;
    } else if (idx < N) {
        output[idx] = gelu_f16_scalar(input[idx]);
    }
}

// OP_TYPE = 4: Float16 Vectorized (8 elements, unpacked)
__global__ void gelu_f16_vec8_unpack_kernel(const half* input, half* output, int N) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 8;
    
    if (idx + 7 < N) {
        // Load 8 half values using vectorized loads (4 half2s or 2 float4s)
        float4 data = *reinterpret_cast<const float4*>(&input[idx]);
        half* h_in = reinterpret_cast<half*>(&data);
        
        half h_out[8];
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            h_out[i] = gelu_f16_scalar(h_in[i]);
        }
        
        // Store result
        *reinterpret_cast<float4*>(&output[idx]) = *reinterpret_cast<float4*>(h_out);
    } else if (idx < N) {
        // Handle remaining elements
        for (int i = idx; i < N; i++) {
            output[i] = gelu_f16_scalar(input[i]);
        }
    }
}

// OP_TYPE = 5: Float16 Vectorized (8 elements, packed half2)
__global__ void gelu_f16_vec8_pack_kernel(const half* input, half* output, int N) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 8;
    
    if (idx + 7 < N) {
        // Load using float4 (8 half elements)
        float4 data = *reinterpret_cast<const float4*>(&input[idx]);
        half2* h2_in = reinterpret_cast<half2*>(&data);
        
        half2 h2_out[4];
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            h2_out[i] = gelu_f16_half2(h2_in[i]);
        }
        
        // Store result
        *reinterpret_cast<float4*>(&output[idx]) = *reinterpret_cast<float4*>(h2_out);
    } else if (idx < N) {
        // Handle remaining elements
        for (int i = idx; i < N; i++) {
            output[i] = gelu_f16_scalar(input[i]);
        }
    }
}

// =============== Main Kernel Entry Point ===============

extern "C" void cuda_kernel(
    void* input,
    void* output,
    int N,
    int op_type
) {
    const int threads = 256;
    int blocks;
    
    switch (op_type) {
        case 0: // Float32 scalar
            blocks = (N + threads - 1) / threads;
            gelu_f32_scalar_kernel<<<blocks, threads>>>(
                reinterpret_cast<const float*>(input),
                reinterpret_cast<float*>(output),
                N
            );
            break;
            
        case 1: // Float32 vec4
            blocks = (N / 4 + threads - 1) / threads;
            gelu_f32_vec4_kernel<<<blocks, threads>>>(
                reinterpret_cast<const float*>(input),
                reinterpret_cast<float*>(output),
                N
            );
            break;
            
        case 2: // Float16 scalar
            blocks = (N + threads - 1) / threads;
            gelu_f16_scalar_kernel<<<blocks, threads>>>(
                reinterpret_cast<const half*>(input),
                reinterpret_cast<half*>(output),
                N
            );
            break;
            
        case 3: // Float16 vec2
            blocks = (N / 2 + threads - 1) / threads;
            gelu_f16_vec2_kernel<<<blocks, threads>>>(
                reinterpret_cast<const half*>(input),
                reinterpret_cast<half*>(output),
                N
            );
            break;
            
        case 4: // Float16 vec8 unpack
            blocks = (N / 8 + threads - 1) / threads;
            gelu_f16_vec8_unpack_kernel<<<blocks, threads>>>(
                reinterpret_cast<const half*>(input),
                reinterpret_cast<half*>(output),
                N
            );
            break;
            
        case 5: // Float16 vec8 pack
            blocks = (N / 8 + threads - 1) / threads;
            gelu_f16_vec8_pack_kernel<<<blocks, threads>>>(
                reinterpret_cast<const half*>(input),
                reinterpret_cast<half*>(output),
                N
            );
            break;
            
        default:
            // Invalid op_type, do nothing
            break;
    }
}
