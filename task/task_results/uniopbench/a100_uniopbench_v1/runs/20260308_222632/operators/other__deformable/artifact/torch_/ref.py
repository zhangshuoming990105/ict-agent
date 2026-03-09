import torch

import MultiScaleDeformableAttention as MSDA


def torch_kernel(
    value_batched,
    value_spatial_shapes_long,
    level_start_index_long,
    sampling_locations_reshaped,
    attention_weights_reshaped,
):
    """
    高性能CUDA实现：纯计算逻辑，无reshape操作

    Args:
        value_batched: (N_, S_, M_, D_) - batched value tensor
        value_spatial_shapes_long: (L_, 2) - spatial shapes (long type)
        level_start_index_long: (L_,) - level start indices (long type)
        sampling_locations_reshaped: (N_, Lq_, M_, L_, P_, 2) - reshaped sampling locations
        attention_weights_reshaped: (N_, Lq_, M_, L_, P_) - reshaped attention weights

    Returns:
        output: (N_, Lq_, M_*D_) - raw output from MSDeformAttnFunction
    """

    im2col_step = 1  # 通常设为1，可以根据GPU内存调整为更大值如2、4等
    output = MSDA.ms_deform_attn_forward(
        value_batched,
        value_spatial_shapes_long,
        level_start_index_long,
        sampling_locations_reshaped,
        attention_weights_reshaped,
        im2col_step,
    )

    return output
