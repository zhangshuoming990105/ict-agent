import os, glob, torch
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension
import torch_npu
from torch_npu.utils.cpp_extension import NpuExtension
NPU_PATH = os.path.dirname(os.path.abspath(torch_npu.__file__))
ext = NpuExtension(
    name="custom_ext",
    sources=glob.glob("./csrc/*.cpp"),
    extra_compile_args=['-I' + os.path.join(NPU_PATH, "include/third_party/acl/inc")],
)
setup(name="custom_op", version='1.0', ext_modules=[ext],
      packages=find_packages(),
      cmdclass={"build_ext": BuildExtension.with_options(use_ninja=False)})
