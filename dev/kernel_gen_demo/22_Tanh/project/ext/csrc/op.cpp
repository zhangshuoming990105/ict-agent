#include <torch/extension.h>
#include <torch/library.h>
#include "pytorch_npu_helper.hpp"

at::Tensor tanh_custom_impl_npu(const at::Tensor& x) {
    at::Tensor y = at::empty_like(x);
    EXEC_NPU_CMD(aclnnTanhCustom, x, y);
    return y;
}

TORCH_LIBRARY(myops, m) {
    m.def("tanh_custom(Tensor x) -> Tensor");
}
TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("tanh_custom", &tanh_custom_impl_npu);
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("tanh_custom", &tanh_custom_impl_npu, "tanh_custom(x) -> tanh(x) (NPU)");
}