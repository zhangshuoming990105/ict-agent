# Skill Level 0: Ascend C 自定义算子工程结构

## 一个完整的 Ascend C 自定义算子由 5 个文件组成

### 1. `op_name.json` — 算子定义
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
- `op` 字段是 PascalCase 算子名，必须与所有其他文件中的注册名一致
- input/output 的 name 与 kernel 的 GM_ADDR 参数、op.cpp 的函数参数一一对应

### 2. `op_host/xxx_tiling.h` — Tiling 数据结构（host ↔ device 共享）
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
- 字段类型只能是 uint32_t, int32_t, float 等 POD 类型
- 类名 `AddCustomTilingData` 和 `REGISTER_TILING_DATA_CLASS` 的第一个参数必须与 op 名一致

### 3. `op_host/xxx.cpp` — 宿主端（tiling + 注册）
三个部分：
- **namespace optiling**: `TilingFunc` — 根据输入 shape 计算 blockDim（多少个 AI Core）和 tileSize（每个 tile 处理多少元素）
- **namespace ge**: `InferShape`（输出 shape = 输入 shape）和 `InferDataType`（输出 dtype = 输入 dtype）
- **namespace ops**: `OP_ADD(AddCustom)` 注册，声明 Input/Output，SetTiling，`AddConfig("ascend910b")`

### 4. `op_kernel/xxx.cpp` — 设备端 kernel（跑在 AI Core 上）
- 入口函数签名固定：`extern "C" __global__ __aicore__ void add_custom(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling)`
- 用 `GET_TILING_DATA(tiling_data, tiling)` 读取 tiling 参数
- 实现 CopyIn → Compute → CopyOut 三级流水线

### 5. `CppExtension/csrc/op.cpp` — PyTorch 封装
- `EXEC_NPU_CMD(aclnnAddCustom, self, other, result)` 调用 kernel
- `PYBIND11_MODULE` 导出 Python 函数

## 编译流程
```bash
msopgen gen -i op.json -c ai_core-ascend910b1 -lan cpp -f aclnn -out ./project
# 替换模板文件
bash build.sh   # 产出 build_out/op_api/lib/libcust_opapi.so
```

## 运行
```bash
ASCEND_CUSTOM_OPP_PATH=./project/build_out python3 setup.py build_ext --inplace
```
