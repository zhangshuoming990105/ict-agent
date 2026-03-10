#include <cuda_runtime.h>
#include <cstdio>

// Compute IoU between two boxes
// Box format: [x1, y1, x2, y2]
__device__ __forceinline__ float compute_iou(const float* box1, const float* box2) {
    // Intersection area
    float inter_x1 = fmaxf(box1[0], box2[0]);
    float inter_y1 = fmaxf(box1[1], box2[1]);
    float inter_x2 = fminf(box1[2], box2[2]);
    float inter_y2 = fminf(box1[3], box2[3]);
    
    float inter_w = fmaxf(0.0f, inter_x2 - inter_x1);
    float inter_h = fmaxf(0.0f, inter_y2 - inter_y1);
    float inter_area = inter_w * inter_h;
    
    // Union area
    float area1 = (box1[2] - box1[0]) * (box1[3] - box1[1]);
    float area2 = (box2[2] - box2[0]) * (box2[3] - box2[1]);
    float union_area = area1 + area2 - inter_area;
    
    return (union_area > 0.0f) ? (inter_area / union_area) : 0.0f;
}

// Optimized NMS kernel using shared memory for reference boxes
__global__ void nms_kernel_optimized(const float* boxes, int* keep, int num_boxes, 
                                      float iou_threshold, int ref_start, int ref_count) {
    // Shared memory to cache reference boxes
    __shared__ float shared_boxes[64][4];  // Cache up to 64 reference boxes
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Early exit for out-of-bounds or already suppressed boxes
    if (idx >= num_boxes) return;
    
    // Load reference boxes into shared memory collaboratively
    int ref_end = min(ref_start + ref_count, num_boxes);
    int actual_ref_count = ref_end - ref_start;
    
    // Each thread helps load reference boxes
    for (int i = threadIdx.x; i < actual_ref_count * 4; i += blockDim.x) {
        int ref_idx = i / 4;
        int coord_idx = i % 4;
        shared_boxes[ref_idx][coord_idx] = boxes[(ref_start + ref_idx) * 4 + coord_idx];
    }
    __syncthreads();
    
    // Check if current box is already suppressed
    if (keep[idx] == 0) return;
    
    // Load current box
    float current_box[4];
    for (int i = 0; i < 4; i++) {
        current_box[i] = boxes[idx * 4 + i];
    }
    
    // Check against all reference boxes in shared memory
    for (int ref_idx = 0; ref_idx < actual_ref_count; ref_idx++) {
        int global_ref_idx = ref_start + ref_idx;
        
        // Only compare with earlier boxes (already processed)
        if (global_ref_idx >= idx) break;
        
        // Skip if reference box is already suppressed
        if (keep[global_ref_idx] == 0) continue;
        
        float iou = compute_iou(shared_boxes[ref_idx], current_box);
        
        if (iou > iou_threshold) {
            keep[idx] = 0;
            return;
        }
    }
}

extern "C" void cuda_kernel(float* boxes, int* keep, int num_boxes, float iou_threshold) {
    const int threads_per_block = 256;
    const int ref_batch_size = 64;  // Process 64 reference boxes per batch
    
    int num_blocks = (num_boxes + threads_per_block - 1) / threads_per_block;
    
    // Process in batches to fit reference boxes in shared memory
    for (int ref_start = 0; ref_start < num_boxes; ref_start += ref_batch_size) {
        nms_kernel_optimized<<<num_blocks, threads_per_block>>>(
            boxes, keep, num_boxes, iou_threshold, ref_start, ref_batch_size
        );
        
        // Synchronize between batches to ensure keep flags are updated
        cudaDeviceSynchronize();
    }
}
