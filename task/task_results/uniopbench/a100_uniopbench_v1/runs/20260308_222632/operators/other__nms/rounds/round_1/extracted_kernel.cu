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

// Block-level NMS kernel processing multiple reference boxes
__global__ void nms_kernel_batch(const float* boxes, int* keep, int num_boxes, 
                                  float iou_threshold, int ref_start, int ref_end) {
    // Grid-stride loop over comparison boxes
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= num_boxes) return;
    if (keep[idx] == 0) return;
    
    const float4* boxes4 = reinterpret_cast<const float4*>(boxes);
    float4 current_box = boxes4[idx];
    
    // Check against all reference boxes in the batch
    for (int ref_idx = ref_start; ref_idx < ref_end && ref_idx < idx; ref_idx++) {
        if (keep[ref_idx] == 0) continue;
        
        float4 ref_box = boxes4[ref_idx];
        float iou = compute_iou(ref_box, current_box);
        
        if (iou > iou_threshold) {
            keep[idx] = 0;
            return;
        }
    }
}

extern "C" void cuda_kernel(float* boxes, int* keep, int num_boxes, float iou_threshold) {
    const int threads_per_block = 256;
    const int batch_size = 64; // Process multiple reference boxes before sync
    
    int num_blocks = (num_boxes + threads_per_block - 1) / threads_per_block;
    
    for (int ref_start = 0; ref_start < num_boxes; ref_start += batch_size) {
        int ref_end = min(ref_start + batch_size, num_boxes);
        
        nms_kernel_batch<<<num_blocks, threads_per_block>>>(
            boxes, keep, num_boxes, iou_threshold, ref_start, ref_end
        );
        
        cudaDeviceSynchronize();
    }
}
