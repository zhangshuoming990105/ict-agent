# Skill Level 2: 完整示例 — AddCustom (两输入 element-wise 加法)

## PyTorch Reference
```python
# torch.add(x, y) -> z，shape 相同
x = torch.randn(16, 1024)
y = torch.randn(16, 1024)
z = x + y
```

## add_custom.json
```json
[{"op": "AddCustom", "language": "cpp",
  "input_desc": [
    {"name": "x", "param_type": "required", "format": ["ND"], "type": ["float"]},
    {"name": "y", "param_type": "required", "format": ["ND"], "type": ["float"]}
  ],
  "output_desc": [
    {"name": "z", "param_type": "required", "format": ["ND"], "type": ["float"]}
  ]}]
```

## add_custom_tiling.h
```cpp
#include "register/tilingdata_base.h"
namespace optiling {
BEGIN_TILING_DATA_DEF(AddCustomTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, totalLength);
  TILING_DATA_FIELD_DEF(uint32_t, tileSize);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(AddCustom, AddCustomTilingData)
}
```

## op_host/add_custom.cpp
```cpp
#include "add_custom_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context) {
    AddCustomTilingData tiling;
    uint32_t totalLength = static_cast<uint32_t>(
        context->GetInputShape(0)->GetOriginShape().GetShapeSize());
    uint32_t blockDim = 8;
    if (totalLength > blockDim * 8192) blockDim = 24;
    if (totalLength > blockDim * 8192) blockDim = 48;
    context->SetBlockDim(blockDim);
    uint32_t tileSize = 8192;
    tiling.set_totalLength(totalLength);
    tiling.set_tileSize(tileSize);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    size_t* ws = context->GetWorkspaceSizes(1);
    ws[0] = 0;
    return ge::GRAPH_SUCCESS;
}
}

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context) {
    *context->GetOutputShape(0) = *context->GetInputShape(0);
    return GRAPH_SUCCESS;
}
static ge::graphStatus InferDataType(gert::InferDataTypeContext* context) {
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return ge::GRAPH_SUCCESS;
}
}

namespace ops {
class AddCustom : public OpDef {
public:
    explicit AddCustom(const char* name) : OpDef(name) {
        this->Input("x").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Input("y").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Output("z").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};
OP_ADD(AddCustom);
}
```

## op_kernel/add_custom.cpp
```cpp
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 2;

class KernelAddCustom {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z,
                                uint32_t totalLength, uint32_t tileSize) {
        uint32_t blockNum = AscendC::GetBlockNum();
        uint32_t blockIdx = AscendC::GetBlockIdx();
        uint32_t chunk = (totalLength + blockNum - 1) / blockNum;
        uint32_t start = blockIdx * chunk;
        coreLen_ = (start >= totalLength) ? 0 : ((totalLength - start < chunk) ? totalLength - start : chunk);
        tileSize_ = tileSize;

        xGm_.SetGlobalBuffer((__gm__ DTYPE_X*)x + start, coreLen_);
        yGm_.SetGlobalBuffer((__gm__ DTYPE_X*)y + start, coreLen_);
        zGm_.SetGlobalBuffer((__gm__ DTYPE_Y*)z + start, coreLen_);

        uint32_t bytes = ((tileSize * sizeof(DTYPE_X)) + 31) & ~31;
        pipe_.InitBuffer(inQueueX_, BUFFER_NUM, bytes);
        pipe_.InitBuffer(inQueueY_, BUFFER_NUM, bytes);
        pipe_.InitBuffer(outQueueZ_, BUFFER_NUM, bytes);
    }

    __aicore__ inline void Process() {
        if (coreLen_ == 0) return;
        uint32_t tiles = (coreLen_ + tileSize_ - 1) / tileSize_;
        for (uint32_t t = 0; t < tiles; ++t) {
            CopyIn(t);
            Compute(t);
            CopyOut(t);
        }
    }

private:
    __aicore__ inline uint32_t CurLen(uint32_t t) {
        uint32_t rem = coreLen_ - t * tileSize_;
        return (rem >= tileSize_) ? tileSize_ : rem;
    }

    __aicore__ inline void CopyIn(uint32_t t) {
        uint32_t len = CurLen(t);
        auto xLocal = inQueueX_.AllocTensor<DTYPE_X>();
        auto yLocal = inQueueY_.AllocTensor<DTYPE_X>();
        AscendC::DataCopy(xLocal, xGm_[t * tileSize_], len);
        AscendC::DataCopy(yLocal, yGm_[t * tileSize_], len);
        inQueueX_.EnQue(xLocal);
        inQueueY_.EnQue(yLocal);
    }

    __aicore__ inline void Compute(uint32_t t) {
        uint32_t len = CurLen(t);
        auto xLocal = inQueueX_.DeQue<DTYPE_X>();
        auto yLocal = inQueueY_.DeQue<DTYPE_X>();
        auto zLocal = outQueueZ_.AllocTensor<DTYPE_Y>();
        AscendC::Add(zLocal, xLocal, yLocal, len);
        outQueueZ_.EnQue<DTYPE_Y>(zLocal);
        inQueueX_.FreeTensor(xLocal);
        inQueueY_.FreeTensor(yLocal);
    }

    __aicore__ inline void CopyOut(uint32_t t) {
        uint32_t len = CurLen(t);
        auto zLocal = outQueueZ_.DeQue<DTYPE_Y>();
        AscendC::DataCopy(zGm_[t * tileSize_], zLocal, len);
        outQueueZ_.FreeTensor(zLocal);
    }

    AscendC::TPipe pipe_;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueX_, inQueueY_;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueZ_;
    AscendC::GlobalTensor<DTYPE_X> xGm_, yGm_;
    AscendC::GlobalTensor<DTYPE_Y> zGm_;
    uint32_t coreLen_, tileSize_;
};

extern "C" __global__ __aicore__ void add_custom(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(td, tiling);
    KernelAddCustom op;
    op.Init(x, y, z, td.totalLength, td.tileSize);
    op.Process();
}
```

## CppExtension/csrc/op.cpp
```cpp
#include <torch/extension.h>
#include <torch/library.h>
#include "pytorch_npu_helper.hpp"

at::Tensor add_custom_impl_npu(const at::Tensor& x, const at::Tensor& y) {
    at::Tensor z = at::empty_like(x);
    EXEC_NPU_CMD(aclnnAddCustom, x, y, z);
    return z;
}

TORCH_LIBRARY(myops, m) {
    m.def("add_custom(Tensor x, Tensor y) -> Tensor");
}
TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("add_custom", &add_custom_impl_npu);
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("add_custom", &add_custom_impl_npu, "add_custom(x, y) -> x+y (NPU)");
}
```

## 关键区别：一输入 vs 二输入
- 一输入（Tanh, Sigmoid, ReLU）：1 个 inQueue + 1 个 outQueue
- 二输入（Add, Mul, Sub）：2 个 inQueue + 1 个 outQueue
- op_json 多一个 input_desc，kernel 多一个 GM_ADDR 参数和 GlobalTensor
