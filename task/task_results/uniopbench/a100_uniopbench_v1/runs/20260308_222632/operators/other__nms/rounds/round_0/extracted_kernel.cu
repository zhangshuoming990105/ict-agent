#include <cuda_runtime.h>
#include <cstdio>

// Compute IoU between two boxes
// Box format: [x1, y1, x2, y2]
__device__ __forceinline__ float compute_iou(const float4& box1, const float4& box2) {
    // Intersection area
    float inter_x1 = fmaxf(box1.x, box2.x);
    float inter_y1 = fmaxf(box1.y, box2.y);
    float inter_x2 = fminf(box1.z, box2.z);
    float inter_y2 = fminf(box1.w, box2.w);
    
    float inter_w = fmaxf(0.0f, inter_x2 - inter_x1);
    float inter_h = fmaxf(0.0f, inter_y2 - inter_y1);
    float inter_area = inter_w * inter_h;
    
    // Union area
    float area1 = (box1.z - box1.x) * (box1.w - box1.y);
    float area2 = (box2.z - box2.x) * (box2.w - box2.y);
    float union_area = area1 + area2 - inter_area;
    
    return (union_area > 0.0f) ? (inter_area / union_area) : 0.0f;
}

// Parallel NMS kernel using CUDA streams
__global__ void nms_kernel_parallel(const float* boxes, int* keep, int num_boxes, 
                                     float iou_threshold, int ref_idx) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x + ref_idx + 1;
    
    if (idx >= num_boxes) return;
    
    // If current box is already suppressed, skip
    if (keep[idx] == 0) return;
    
    // If reference box is suppressed, skip
    if (keep[ref_idx] == 0) return;
    
    // Load boxes using vectorized loads
    const float4* boxes4 = reinterpret_cast<const float4*>(boxes);
    float4 ref_box = boxes4[ref_idx];
    float4 current_box = boxes4[idx];
    
    float iou = compute_iou(ref_box, current_box);
    
    // Suppress if IoU exceeds threshold
    if (iou > iou_threshold) {
        keep[idx] = 0;
    }
}

extern "C" void cuda_kernel(float* boxes, int* keep, int num_boxes, float iou_threshold) {
    const int threads_per_block = 256;
    const int streams_count = 8;
    
    // Create streams for concurrent kernel execution
    cudaStream_t streams[streams_count];
    for (int i = 0; i < streams_count; i++) {
        cudaStreamCreate(&streams[i]);
    }
    
    // Process each box sequentially as a reference
    for (int ref_idx = 0; ref_idx < num_boxes; ref_idx++) {
        int remaining = num_boxes - ref_idx - 1;
        if (remaining <= 0) break;
        
        int num_blocks = (remaining + threads_per_block - 1) / threads_per_block;
        
        // Use round-robin stream assignment
        int stream_idx = ref_idx % streams_count;
        
        nms_kernel_parallel<<<num_blocks, threads_per_block, 0, streams[stream_idx]>>>(
            boxes, keep, num_boxes, iou_threshold, ref_idx
        );
        
        // Synchronize periodically to ensure correctness
        if ((ref_idx + 1) % (streams_count * 2) == 0) {
            cudaDeviceSynchronize();
        }
    }
    
    // Final synchronization
    cudaDeviceSynchronize();
    
    // Cleanup streams
    for (int i = 0; i < streams_count; i++) {
        cudaStreamDestroy(streams[i]);
    }
}
