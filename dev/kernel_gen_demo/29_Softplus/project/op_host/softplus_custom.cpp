#include "softplus_custom_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context) {
    SoftplusCustomTilingData tiling;
    uint32_t totalLength = static_cast<uint32_t>(
        context->GetInputShape(0)->GetOriginShape().GetShapeSize());

    // With softplus we need: inQueue(2 buffers) + outQueue(2 buffers) + tmpQueue(2 buffers)
    // That's 6 * tileSize * 4 bytes. UB ~ 192KB = 196608 bytes
    // 6 * tileSize * 4 <= 196608 => tileSize <= 8192
    // Use 4096 to be safe
    uint32_t tileSize = 4096;

    uint32_t blockDim = 48;
    // Make sure each core has at least some work
    uint32_t elemsPerCore = (totalLength + blockDim - 1) / blockDim;
    // Ensure elemsPerCore is aligned to tileSize for simplicity
    // but it's fine if it's not, the kernel handles remainder

    context->SetBlockDim(blockDim);
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
class SoftplusCustom : public OpDef {
public:
    explicit SoftplusCustom(const char* name) : OpDef(name) {
        this->Input("x").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Output("y").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};
OP_ADD(SoftplusCustom);
}