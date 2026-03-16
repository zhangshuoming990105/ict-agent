# Skill Level 1: Ascend C Kernel 编程 — Element-wise 算子

## AI Core 编程模型

Ascend AI Core 有三级流水线：
1. **MTE2 (Memory Transfer Engine)**: Global Memory → Local Memory (UB)
2. **Vector/Scalar Unit**: 在 UB 上做计算
3. **MTE3**: UB → Global Memory

用 `TPipe` + `TQue` 实现双缓冲流水线，CopyIn 和 Compute 可以 overlap。

## Kernel 类模板

```cpp
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 2;  // 双缓冲

class KernelXxxCustom {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y,
                                uint32_t totalLength, uint32_t tileSize) {
        // 多核分片
        uint32_t blockNum = AscendC::GetBlockNum();
        uint32_t blockIdx = AscendC::GetBlockIdx();
        uint32_t coreChunk = (totalLength + blockNum - 1) / blockNum;
        uint32_t coreStart = blockIdx * coreChunk;
        coreLen_ = min(coreChunk, totalLength - coreStart);

        xGm_.SetGlobalBuffer((__gm__ DTYPE_X*)x + coreStart, coreLen_);
        yGm_.SetGlobalBuffer((__gm__ DTYPE_Y*)y + coreStart, coreLen_);

        // UB 分配（必须 32B 对齐）
        uint32_t xBytes = ((tileSize * sizeof(DTYPE_X)) + 31) & ~31;
        pipe_.InitBuffer(inQueueX_, BUFFER_NUM, xBytes);
        pipe_.InitBuffer(outQueueY_, BUFFER_NUM, xBytes);
    }

    __aicore__ inline void Process() {
        uint32_t tiles = (coreLen_ + tileSize_ - 1) / tileSize_;
        for (uint32_t t = 0; t < tiles; ++t) {
            CopyIn(t);
            Compute(t);
            CopyOut(t);
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t t) {
        uint32_t len = min(tileSize_, coreLen_ - t * tileSize_);
        auto xLocal = inQueueX_.AllocTensor<DTYPE_X>();
        AscendC::DataCopy(xLocal, xGm_[t * tileSize_], len);
        inQueueX_.EnQue(xLocal);
    }

    __aicore__ inline void Compute(uint32_t t) {
        uint32_t len = min(tileSize_, coreLen_ - t * tileSize_);
        auto xLocal = inQueueX_.DeQue<DTYPE_X>();
        auto yLocal = outQueueY_.AllocTensor<DTYPE_Y>();

        // === 这里放实际的计算 ===
        AscendC::Adds(yLocal, xLocal, 1.0f, len);  // 示例：y = x + 1

        outQueueY_.EnQue<DTYPE_Y>(yLocal);
        inQueueX_.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(uint32_t t) {
        uint32_t len = min(tileSize_, coreLen_ - t * tileSize_);
        auto yLocal = outQueueY_.DeQue<DTYPE_Y>();
        AscendC::DataCopy(yGm_[t * tileSize_], yLocal, len);
        outQueueY_.FreeTensor(yLocal);
    }

    AscendC::TPipe pipe_;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueX_;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueY_;
    AscendC::GlobalTensor<DTYPE_X> xGm_;
    AscendC::GlobalTensor<DTYPE_Y> yGm_;
    uint32_t coreLen_, tileSize_;
};
```

## 可用的 AscendC 向量计算 API

### 一元运算（dst, src, count）
| API | 数学 | 需要 tmpBuf |
|-----|------|------------|
| `Abs(dst, src, count)` | \|x\| | 否 |
| `Neg(dst, src, count)` | -x | 否 |
| `Exp(dst, src, count)` | e^x | 否 |
| `Relu(dst, src, count)` | max(0,x) | 否 |
| `Sigmoid(dst, src, tmp, count)` | 1/(1+e^-x) | 是 |
| `Tanh(dst, src, tmp, count)` | tanh(x) | 是 |
| `Ln(dst, src, count)` | ln(x) | 否 |
| `Reciprocal(dst, src, count)` | 1/x | 否 |
| `Sqrt(dst, src, count)` | sqrt(x) | 否 |
| `Rsqrt(dst, src, count)` | 1/sqrt(x) | 否 |

### 标量运算（dst, src, scalar, count）
| API | 数学 |
|-----|------|
| `Adds(dst, src, scalar, count)` | x + s |
| `Muls(dst, src, scalar, count)` | x * s |
| `Mins(dst, src, scalar, count)` | min(x, s) |
| `Maxs(dst, src, scalar, count)` | max(x, s) |

### 二元运算（dst, src1, src2, count）
| API | 数学 |
|-----|------|
| `Add(dst, src1, src2, count)` | x + y |
| `Sub(dst, src1, src2, count)` | x - y |
| `Mul(dst, src1, src2, count)` | x * y |
| `Div(dst, src1, src2, count)` | x / y |
| `Min(dst, src1, src2, count)` | min(x, y) |
| `Max(dst, src1, src2, count)` | max(x, y) |

### 需要 tmpBuf 的运算
Sigmoid、Tanh 等需要一个额外的 `LocalTensor<uint8_t> tmpBuf`（注意 Ln 不需要）：
```cpp
// 分配 tmp buffer
pipe_.InitBuffer(tmpQueue_, 1, tmpSizeBytes);
auto tmpLocal = tmpQueue_.AllocTensor<uint8_t>();

// 使用
AscendC::Tanh(yLocal, xLocal, tmpLocal, len);

// 释放
tmpQueue_.FreeTensor(tmpLocal);
```
tmpSizeBytes 经验值：`1024 + 2 * tileSize`，向上对齐到 32B。

## 组合实现复杂函数

没有直接 API 的函数需要用基础 API 组合：

**GELU**: `x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))`
```cpp
// 简化近似：用 Sigmoid 近似
// gelu(x) ≈ x * sigmoid(1.702 * x)
AscendC::Muls(tmp1, xLocal, 1.702f, len);
AscendC::Sigmoid(tmp2, tmp1, tmpBuf, len);
AscendC::Mul(yLocal, xLocal, tmp2, len);
```

**Swish/SiLU**: `x * sigmoid(x)`
```cpp
AscendC::Sigmoid(tmp, xLocal, tmpBuf, len);
AscendC::Mul(yLocal, xLocal, tmp, len);
```

**Softplus**: `ln(1 + exp(x))`
```cpp
AscendC::Exp(tmp, xLocal, len);
AscendC::Adds(tmp, tmp, 1.0f, len);
AscendC::Ln(yLocal, tmp, len);
```

**HardTanh**: `clamp(x, -1, 1)`
```cpp
AscendC::Maxs(tmp, xLocal, -1.0f, len);
AscendC::Mins(yLocal, tmp, 1.0f, len);
```

**HardSigmoid**: `clamp((x + 3) / 6, 0, 1)`
```cpp
AscendC::Adds(tmp, xLocal, 3.0f, len);
AscendC::Muls(tmp, tmp, 1.0f/6.0f, len);
AscendC::Maxs(tmp, tmp, 0.0f, len);
AscendC::Mins(yLocal, tmp, 1.0f, len);
```

## Tiling 策略建议（element-wise）

- `blockDim`: min(48, ceil(totalLength / 4096))，至少保证每个核有 4096 个元素
- `tileSize`: 8192 元素（fp32 = 32KB），适合 UB 大小
- 如果需要 tmpBuf，减小 tileSize 到 4096
- UB 总预算 ~192KB，双缓冲 x/y 各 2 份 = 4 * tileSize * 4B
