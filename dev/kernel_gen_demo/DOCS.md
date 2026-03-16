# EvoKernel-lite: LLM 自动生成 Ascend C Kernel — 详细文档

## 1. 系统概述

本系统从 PyTorch reference 算子出发，调用 LLM 自动生成可在 Ascend 910B NPU 上运行的 Ascend C 自定义算子，并通过编译-测试-反馈循环迭代修正。

### 对标 EvoKernel 论文


| 论文 EvoKernel                                                      | 本 Demo                                   |
| ----------------------------------------------------------------- | ---------------------------------------- |
| Memory-based MDP + Value-Driven Retrieval                         | Skills 文件作为 system prompt                |
| Multi-gate Verifier (anti-hack + compile + correctness + latency) | compile → extension → correctness + perf |
| Cold-Start Drafting (binary feasibility reward, 重试)               | 错误信息反馈 + LLM 重试（max_attempts 轮）          |
| Continual Refining (latency 优化)                                   | 未实现（可扩展）                                 |
| Cross-task memory sharing                                         | Skills 中的 Add 示例跨任务复用                    |


### 数据流

```
PyTorch Reference (.py)
        │
        ▼
┌─────────────────┐
│  Skills (L0-L2) │──→ System Prompt
└─────────────────┘
        │
        ▼
┌─────────────────┐
│   Claude LLM    │──→ 生成 5 个文件
└─────────────────┘
        │
        ▼
┌─────────────────┐      ┌──────────────┐
│  msopgen + opc  │──→   │ build.sh     │──→ libcust_opapi.so
└─────────────────┘      └──────────────┘
        │
        ▼
┌─────────────────┐
│  CppExtension   │──→ custom_ext.so
└─────────────────┘
        │
        ▼
┌─────────────────┐
│  子进程测试       │──→ correctness + latency
└─────────────────┘
        │
        ├── PASS → 结束
        └── FAIL → 带错误反馈回到 LLM
```

## 2. Skills 层级

Skills 是给 LLM 的知识文档，分三层：


| 文件                        | 层级      | 内容                                   |
| -------------------------- | --------- | ------------------------------------- |
| `L0_project_structure.md`  | 工程结构   | 5 个文件的作用、编译流程、运行方式         |
| `L1_elementwise_kernel.md` | API 参考  | AI Core 编程模型、向量 API 列表、组合实现  |
| `L2_example_add.md`        | 完整示例   | Add 算子的 5 个文件完整代码              |


LLM 收到的 system prompt = Header + L0 + L1 + L2。

## 3. 逐函数解释

### 3.1 配置与数据结构

```python
HELPER_HPP = Path("/workspace/dev/ascend-templates/cpp_extension/pytorch_npu_helper.hpp")
```

`pytorch_npu_helper.hpp` 是来自 `ascend/samples` 官方仓库的辅助头文件，提供 `EXEC_NPU_CMD` 宏。

```python
SKILLS_DIR = Path("/workspace/dev/kernel_gen_demo/skills")
```

Skills 文件目录，包含 L0-L2 三个 markdown 文件。

```python
@dataclass
class Attempt:
    round: int                      # 第几轮
    compiled: bool = False          # msopgen + build.sh 是否成功
    compile_error: str = ""         # 编译错误信息
    extension_built: bool = False   # CppExtension 是否构建成功
    extension_error: str = ""
    correctness: bool | None = None # 正确性是否通过
    max_diff: float | None = None   # 与 torch 参考的最大绝对误差
    latency_ms: float | None = None # 自定义 kernel 延迟
    ref_latency_ms: float | None = None # torch 参考延迟
    speedup: float | None = None
    error: str = ""
    kernel_code: str = ""           # 保存生成的 kernel 代码
    host_code: str = ""
    tiling_code: str = ""
```

### 3.2 LLM 客户端

```python
def create_client() -> anthropic.Anthropic:
```

从环境变量 `ANTHROPIC_BASE_URL` 和 `ANTHROPIC_AUTH_TOKEN` 创建 Anthropic 客户端。

```python
def load_skills() -> str:
```

读取 `skills/L*.md` 文件，按文件名排序后用 `---` 分隔拼接。

```python
def build_system_prompt() -> str:
```

拼接 header（输出格式说明）+ skills 内容，构成完整的 system prompt。
**关键**：不包含任何目标算子的实现代码，只有 Add 示例和 API 文档。

```python
def call_llm(client, system, user_msg, model=None) -> str:
```

调用 Claude API，返回纯文本响应。model 默认从 `ANTHROPIC_MODEL` 环境变量读取。

### 3.3 代码解析

```python
def parse_generated_code(text: str) -> dict[str, str]:
```

从 LLM 输出中按 `===MARKER_START===` / `===MARKER_END===` 标记提取 5 个文件：

- `op_json`: 算子定义 JSON
- `tiling`: tiling.h 头文件
- `host`: op_host .cpp
- `kernel`: op_kernel .cpp
- `op_cpp`: CppExtension op.cpp

### 3.4 名称推导

```python
def infer_op_name(op_json_str: str) -> str:
```

从 op_json 的 `"op"` 字段提取 PascalCase 算子名（如 `"TanhCustom"`）。

```python
def infer_snake_name(op_name: str) -> str:
```

PascalCase → snake_case：`TanhCustom` → `tanh_custom`。
用正则在大写字母前插入 `_` 然后转小写。

### 3.5 编译流程

```python
def setup_project(work_dir, codes, op_name) -> Path:
```

1. 将 `codes["op_json"]` 写到 `work_dir/<snake>.json`
2. 调用 `msopgen gen -i <json> -c ai_core-ascend910b1 -lan cpp -f aclnn -out project/`
3. 推导 msopgen 生成的 snake_case 名（可能与 LLM 生成的不同）
4. 将 LLM 生成的 kernel/host/tiling 写入 msopgen 的文件名位置
5. 处理 `#include` 的 tiling.h 文件名不匹配

返回 project 目录路径。

```python
def compile_project(project_dir) -> (bool, str):
```

在 project_dir 下执行 `bash build.sh`，检查 `build_out/op_api/lib/libcust_opapi.so` 是否生成。
超时 300 秒。

```python
def build_extension(work_dir, project_dir, codes) -> (bool, str):
```

1. 在 `project_dir/ext/csrc/` 下写入 `op.cpp` 和 `pytorch_npu_helper.hpp`
2. 生成 `setup.py`（使用 `NpuExtension`）
3. 设置 `ASCEND_CUSTOM_OPP_PATH` 后执行 `python3 setup.py build_ext --inplace`
4. 检查 `custom_ext*.so` 是否生成

### 3.6 测试

```python
def run_test_in_subprocess(work_dir, project_dir, ref_path, device) -> dict:
```

**为什么用子进程**：所有 EvoKernel 的 `op.cpp` 都注册到 `TORCH_LIBRARY(myops, m)`，在同一进程中加载多个 extension 会导致命名空间冲突崩溃。

子进程脚本流程：

1. 设 `ASCEND_CUSTOM_OPP_PATH`
2. `import custom_ext`
3. 从 PyTorch reference 加载 `Model` + `get_inputs()`
4. 运行 torch reference 得到 `y_ref`
5. 调用 `ext_func(*tensor_inputs)` 得到 `y_out`
6. `torch.allclose(y_ref, y_out, atol=1e-2)` 检查正确性
7. Warmup + 10 次计时
8. 以 JSON 格式输出结果到 stdout

### 3.7 主循环

```python
def main():
```

1. 解析命令行参数（`--ref`, `--device`, `--max-attempts`, `--model`）
2. 创建 LLM 客户端，加载 skills 构建 system prompt
3. 循环 max_attempts 轮：
  - 第 1 轮：发送 "Generate ... for this PyTorch reference: ..."
  - 后续轮：发送 "Previous attempt failed. Error: ... Please fix."
  - 解析 LLM 输出
  - 编译 → Extension → 测试
  - PASS 则跳出循环
4. 打印汇总结果，保存 `summary.json`

## 4. 运行示例

```bash
cd /workspace/dev/kernel_gen_demo

# 简单 elementwise（1 轮通过）
python3 kernel_gen.py --ref .../22_Tanh.py --device 4

# 需要组合 API（可能多轮）
python3 kernel_gen.py --ref .../26_GELU_.py --device 4 --max-attempts 5
```

## 5. 已验证结果


| 算子       | 轮次  | 正确性             | 加速比   | 说明                   |
| -------- | --- | --------------- | ----- | -------------------- |
| Tanh     | 1/5 | PASS (1.19e-07) | 0.10x | tiling 未优化           |
| Sigmoid  | 1/5 | PASS (0)        | 1.00x | 直接用 AscendC::Sigmoid |
| ReLU     | 1/3 | PASS (0)        | 1.00x | 直接用 AscendC::Relu    |
| GELU     | 5/5 | PASS (1.79e-07) | 0.99x | 组合 API，经历多轮修正        |
| Softplus | 1/5 | PASS (1.19e-07) | 2.48x | Agent 独立完成           |


