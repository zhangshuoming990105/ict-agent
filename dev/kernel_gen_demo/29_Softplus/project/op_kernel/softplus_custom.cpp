#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 2;

class KernelSoftplusCustom {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y,
                                uint32_t totalLength, uint32_t tileSize) {
        uint32_t blockNum = AscendC::GetBlockNum();
        uint32_t blockIdx = AscendC::GetBlockIdx();
        uint32_t chunk = (totalLength + blockNum - 1) / blockNum;
        uint32_t start = blockIdx * chunk;
        coreLen_ = (start >= totalLength) ? 0 : 
                   ((totalLength - start < chunk) ? totalLength - start : chunk);
        tileSize_ = tileSize;

        xGm_.SetGlobalBuffer((__gm__ float*)x + start, coreLen_);
        yGm_.SetGlobalBuffer((__gm__ float*)y + start, coreLen_);

        uint32_t bytes = ((tileSize * sizeof(float)) + 31) & ~31;
        pipe_.InitBuffer(inQueueX_, BUFFER_NUM, bytes);
        pipe_.InitBuffer(outQueueY_, BUFFER_NUM, bytes);
        pipe_.InitBuffer(tmpQueue_, BUFFER_NUM, bytes);
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
        auto xLocal = inQueueX_.AllocTensor<float>();
        AscendC::DataCopy(xLocal, xGm_[t * tileSize_], len);
        inQueueX_.EnQue(xLocal);
    }

    __aicore__ inline void Compute(uint32_t t) {
        uint32_t len = CurLen(t);
        auto xLocal = inQueueX_.DeQue<float>();
        auto yLocal = outQueueY_.AllocTensor<float>();
        auto tmpLocal = tmpQueue_.AllocTensor<float>();

        // softplus(x) = ln(1 + exp(x))
        // Step 1: tmpLocal = exp(x)
        AscendC::Exp(tmpLocal, xLocal, len);
        // Step 2: tmpLocal = 1 + exp(x)
        AscendC::Adds(tmpLocal, tmpLocal, 1.0f, len);
        // Step 3: yLocal = ln(1 + exp(x))
        AscendC::Ln(yLocal, tmpLocal, len);

        outQueueY_.EnQue<float>(yLocal);
        tmpQueue_.FreeTensor(tmpLocal);
        inQueueX_.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(uint32_t t) {
        uint32_t len = CurLen(t);
        auto yLocal = outQueueY_.DeQue<float>();
        AscendC::DataCopy(yGm_[t * tileSize_], yLocal, len);
        outQueueY_.FreeTensor(yLocal);
    }

    AscendC::TPipe pipe_;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueX_;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueY_;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> tmpQueue_;
    AscendC::GlobalTensor<float> xGm_;
    AscendC::GlobalTensor<float> yGm_;
    uint32_t coreLen_, tileSize_;
};

extern "C" __global__ __aicore__ void softplus_custom(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(td, tiling);
    KernelSoftplusCustom op;
    op.Init(x, y, td.totalLength, td.tileSize);
    op.Process();
}