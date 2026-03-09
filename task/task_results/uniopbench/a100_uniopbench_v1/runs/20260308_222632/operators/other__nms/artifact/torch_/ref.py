import torch
import torchvision

def torch_kernel(boxes, iou_threshold):
    num_boxes = boxes.shape[0]
    scores = torch.arange(num_boxes, 0, -1, device=boxes.device, dtype=torch.float32)
    
    keep_indices = torchvision.ops.nms(boxes, scores, iou_threshold)
    
    keep_mask = torch.zeros(num_boxes, dtype=torch.int32, device=boxes.device)
    keep_mask[keep_indices] = 1
    
    return keep_mask