#!/usr/bin/env python
"""Simplified compile script: compile root and kernels CUDA/C++ sources."""

from __future__ import annotations

import os
import re
import shutil
import sys
import traceback
from pathlib import Path

import torch
import torch.utils.cpp_extension as cpp_ext


def detect_backend() -> str:
    if getattr(torch.version, "hip", None):
        return "hip"
    if torch.version.cuda is not None:
        return "cuda"
    return "none"


def _has_hip_counterpart(name: str, all_names: set[str]) -> str | None:
    stem = Path(name).stem
    cand1 = stem + "_hip.cpp"
    if cand1 in all_names:
        return cand1
    if "cuda" in name:
        cand2 = name.replace("cuda", "hip")
        if cand2 in all_names and cand2 != name:
            return cand2
    return None


def find_sources(backend: str) -> list[str]:
    root = Path(".")
    kernels_dir = Path("kernels")
    root_sources = [str(path) for path in root.glob("*.cu")] + [str(path) for path in root.glob("*.cpp")]
    kernel_sources: list[str] = []
    if kernels_dir.is_dir():
        kernel_cu = list(kernels_dir.glob("*.cu"))
        kernel_hip = list(kernels_dir.glob("*.hip"))
        kernel_cpp = list(kernels_dir.glob("*.cpp"))
        if backend == "hip":
            hip_stems = {path.stem for path in kernel_hip}
            for path in kernel_cu:
                if path.stem in hip_stems:
                    kernel_sources.append(str(path.parent / (path.stem + ".hip")))
                else:
                    kernel_sources.append(str(path))
            cu_stems = {path.stem for path in kernel_cu}
            for path in kernel_hip:
                if path.stem not in cu_stems:
                    kernel_sources.append(str(path))
            cpp_names = {path.name: path for path in kernel_cpp}
            all_cpp_names = set(cpp_names.keys())
            skip = set()
            for name in all_cpp_names:
                if name in skip:
                    continue
                hip_name = _has_hip_counterpart(name, all_cpp_names)
                if hip_name:
                    skip.add(name)
            for name, path in cpp_names.items():
                if name not in skip:
                    kernel_sources.append(str(path))
        else:
            selected_cpp = [path for path in kernel_cpp if not path.name.endswith("_hip.cpp")]
            kernel_sources = [str(path) for path in kernel_cu] + [str(path) for path in selected_cpp]
    return sorted(set(root_sources + kernel_sources))


def _patch_shape_to_sizes(sources: list[str]) -> None:
    pattern = re.compile(r"\.shape\b")
    for src in sources:
        path = Path(src)
        if not path.exists() or path.suffix not in (".cpp", ".h", ".cuh"):
            continue
        text = path.read_text(errors="replace")
        if ".shape" not in text:
            continue
        new_text = pattern.sub(".sizes()", text)
        if new_text != text:
            path.write_text(new_text)
            print(f"  [patch] {src}: .shape -> .sizes()")


def compile_kernels() -> int:
    build_dir = Path("build/forced_compile")
    output_so = Path("cuda_extension.so")
    backend = detect_backend()
    sources = find_sources(backend)
    if backend == "none":
        print("Error: this PyTorch build has neither CUDA nor HIP backend.")
        return 1
    if not sources:
        print("Error: no source files found (*.cu, *.cpp in root or kernels/)")
        return 1
    print(f"Backend: {backend}")
    print(f"Compiling {len(sources)} files: {', '.join(sources)}")
    _patch_shape_to_sizes(sources)
    build_dir.mkdir(parents=True, exist_ok=True)
    if output_so.exists():
        output_so.unlink()
    cwd = os.path.abspath(".")
    include_paths = [cwd, os.path.join(cwd, "kernels")]
    try:
        if backend == "hip":
            from torch.utils.hipify import hipify_python

            real_hipify = hipify_python.hipify

            def safe_hipify(**kwargs):
                result = real_hipify(**kwargs)
                for path, hipify_result in result.items():
                    if hipify_result.hipified_path is None:
                        hipify_result.hipified_path = path
                return result

            hipify_python.hipify = safe_hipify
            try:
                module = cpp_ext.load(
                    name="cuda_extension",
                    sources=sources,
                    build_directory=str(build_dir),
                    verbose=True,
                    with_cuda=True,
                    extra_cflags=["-O3", "-std=c++17"],
                    extra_cuda_cflags=["-O3"],
                    extra_include_paths=include_paths,
                )
            finally:
                hipify_python.hipify = real_hipify
        else:
            module = cpp_ext.load(
                name="cuda_extension",
                sources=sources,
                build_directory=str(build_dir),
                verbose=True,
                with_cuda=True,
                extra_cflags=["-O3", "-std=c++17"],
                extra_cuda_cflags=["-O3"],
                extra_include_paths=include_paths,
            )
    except Exception as exc:
        print("Compilation failed.")
        print(str(exc))
        traceback.print_exc()
        return 1
    built_so = Path(module.__file__)
    if built_so.exists():
        shutil.copy2(built_so, output_so)
        print(f"Compile success: {output_so}")
        return 0
    print("Compilation finished but cuda_extension.so was not generated.")
    return 1


def main() -> int:
    return compile_kernels()


if __name__ == "__main__":
    sys.exit(main())
