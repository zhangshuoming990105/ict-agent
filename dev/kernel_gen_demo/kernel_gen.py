#!/usr/bin/env python3
"""
EvoKernel-lite: 用 LLM 自动生成 Ascend C kernel 并验证。

简化版 EvoKernel pipeline：
  1. 读取 PyTorch reference（如 torch.tanh）
  2. 调用 Claude 生成 Ascend C kernel 三件套（op_kernel, op_host, op_host_tiling）
  3. msopgen + build.sh 编译
  4. 构建 CppExtension
  5. 对比 PyTorch reference 验证正确性
  6. 失败则带错误信息重试（最多 N 轮）

用法:
  python3 kernel_gen.py --ref /path/to/22_Tanh.py --device 4
  python3 kernel_gen.py --ref /path/to/21_Sigmoid.py --device 4
  python3 kernel_gen.py --ref /path/to/26_GELU_.py --device 4 --max-attempts 5
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import textwrap
import time
from dataclasses import dataclass, field
from pathlib import Path

import anthropic

# ─── 路径配置 ────────────────────────────────────────────────────────────────
HELPER_HPP = Path("/workspace/dev/ascend-templates/cpp_extension/pytorch_npu_helper.hpp")
CANN_PATH = os.getenv("ASCEND_HOME_PATH", "/usr/local/Ascend/cann-8.5.0")


@dataclass
class Attempt:
    round: int
    compiled: bool = False
    compile_error: str = ""
    extension_built: bool = False
    extension_error: str = ""
    correctness: bool | None = None
    max_diff: float | None = None
    latency_ms: float | None = None
    ref_latency_ms: float | None = None
    speedup: float | None = None
    error: str = ""
    kernel_code: str = ""
    host_code: str = ""
    tiling_code: str = ""


# ─── LLM 调用 ───────────────────────────────────────────────────────────────
SKILLS_DIR = Path("/workspace/dev/kernel_gen_demo/skills")


def create_client() -> anthropic.Anthropic:
    return anthropic.Anthropic(
        base_url=os.environ["ANTHROPIC_BASE_URL"],
        api_key=os.environ["ANTHROPIC_AUTH_TOKEN"],
    )


def load_skills() -> str:
    """读取所有 skill 文件拼接为 system prompt。"""
    parts = []
    for skill_file in sorted(SKILLS_DIR.glob("L*.md")):
        parts.append(skill_file.read_text())
    return "\n\n---\n\n".join(parts)


def build_system_prompt() -> str:
    skills = load_skills()
    header = textwrap.dedent("""\
    You are an expert Ascend C kernel developer for Huawei Ascend 910B NPUs.
    Given a PyTorch reference operator, generate a complete Ascend C custom operator.

    You must output exactly 5 sections in this format:

    ===OP_JSON_START===
    <content>
    ===OP_JSON_END===

    ===TILING_H_START===
    <content>
    ===TILING_H_END===

    ===HOST_CPP_START===
    <content>
    ===HOST_CPP_END===

    ===KERNEL_CPP_START===
    <content>
    ===KERNEL_CPP_END===

    ===OP_CPP_START===
    <content>
    ===OP_CPP_END===

    Below are skill documents describing the project structure, APIs, and a complete Add example.
    Study them carefully, then generate the requested operator following the same patterns.

    """)
    return header + skills


def call_llm(
    client: anthropic.Anthropic,
    system: str,
    user_msg: str,
    model: str | None = None,
) -> str:
    model = model or os.environ.get("ANTHROPIC_MODEL", "mco-4")
    resp = client.messages.create(
        model=model,
        max_tokens=16000,
        system=system,
        messages=[{"role": "user", "content": user_msg}],
    )
    return resp.content[0].text


def parse_generated_code(text: str) -> dict[str, str]:
    """从 LLM 输出中提取各文件内容。"""
    markers = {
        "op_json": ("===OP_JSON_START===", "===OP_JSON_END==="),
        "tiling": ("===TILING_H_START===", "===TILING_H_END==="),
        "host": ("===HOST_CPP_START===", "===HOST_CPP_END==="),
        "kernel": ("===KERNEL_CPP_START===", "===KERNEL_CPP_END==="),
        "op_cpp": ("===OP_CPP_START===", "===OP_CPP_END==="),
    }
    result = {}
    for key, (start, end) in markers.items():
        m = re.search(re.escape(start) + r"\s*\n(.*?)\n\s*" + re.escape(end), text, re.DOTALL)
        if m:
            result[key] = m.group(1).strip()
        else:
            result[key] = ""
    return result


# ─── 编译 & 测试 ────────────────────────────────────────────────────────────
def infer_op_name(op_json_str: str) -> str:
    """从 op_json 中提取算子名。"""
    try:
        data = json.loads(op_json_str)
        if isinstance(data, list):
            return data[0]["op"]
        return data.get("op", "CustomOp")
    except Exception:
        return "CustomOp"


def infer_snake_name(op_name: str) -> str:
    """PascalCase -> snake_case: TanhCustom -> tanh_custom"""
    s = re.sub(r'(?<!^)(?=[A-Z])', '_', op_name).lower()
    return s


def setup_project(work_dir: Path, codes: dict[str, str], op_name: str) -> Path:
    """设置编译项目：msopgen + 替换代码。"""
    snake = infer_snake_name(op_name)
    project_dir = work_dir / "project"

    # 写 op_json
    json_path = work_dir / f"{snake}.json"
    json_path.write_text(codes["op_json"])

    # msopgen gen
    if project_dir.exists():
        shutil.rmtree(project_dir)

    result = subprocess.run(
        ["msopgen", "gen", "-i", str(json_path),
         "-c", "ai_core-ascend910b1", "-lan", "cpp", "-f", "aclnn",
         "-out", str(project_dir)],
        capture_output=True, text=True, timeout=60,
    )
    if result.returncode != 0:
        return project_dir

    # 推导 msopgen 的 snake name
    msopgen_files = list((project_dir / "op_kernel").glob("*.cpp"))
    msopgen_snake = msopgen_files[0].stem if msopgen_files else snake

    # 写入生成的代码
    (project_dir / f"op_kernel/{msopgen_snake}.cpp").write_text(codes["kernel"])
    (project_dir / f"op_host/{msopgen_snake}.cpp").write_text(codes["host"])
    (project_dir / f"op_host/{msopgen_snake}_tiling.h").write_text(codes["tiling"])

    # 修复 include：如果 host.cpp include 的 tiling.h 名字和文件名不同
    host_text = codes["host"]
    m = re.search(r'#include\s+"([^"]*_tiling\.h)"', host_text)
    if m:
        included = m.group(1)
        if included != f"{msopgen_snake}_tiling.h":
            (project_dir / f"op_host/{included}").write_text(codes["tiling"])

    return project_dir


def compile_project(project_dir: Path) -> tuple[bool, str]:
    """编译算子。"""
    try:
        result = subprocess.run(
            ["bash", "build.sh"],
            cwd=str(project_dir),
            capture_output=True, text=True, check=True, timeout=300,
        )
        opapi = project_dir / "build_out/op_api/lib/libcust_opapi.so"
        if opapi.exists():
            return True, ""
        return False, "libcust_opapi.so not found after build"
    except subprocess.CalledProcessError as e:
        return False, e.stderr[-1000:] if e.stderr else e.stdout[-1000:]
    except subprocess.TimeoutExpired:
        return False, "Build timed out (300s)"


def build_extension(work_dir: Path, project_dir: Path, codes: dict[str, str]) -> tuple[bool, str]:
    """构建 PyTorch CppExtension。"""
    ext_dir = project_dir / "ext"
    csrc_dir = ext_dir / "csrc"
    csrc_dir.mkdir(parents=True, exist_ok=True)

    (csrc_dir / "op.cpp").write_text(codes["op_cpp"])
    shutil.copy2(HELPER_HPP, csrc_dir / "pytorch_npu_helper.hpp")

    setup_py = ext_dir / "setup.py"
    setup_py.write_text(textwrap.dedent("""\
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
    """))

    try:
        env = os.environ.copy()
        env["ASCEND_CUSTOM_OPP_PATH"] = str(project_dir / "build_out")
        subprocess.run(
            [sys.executable, "setup.py", "build_ext", "--inplace"],
            cwd=str(ext_dir), capture_output=True, text=True, check=True,
            timeout=180, env=env,
        )
        so_files = list(ext_dir.glob("custom_ext*.so"))
        if so_files:
            return True, ""
        return False, "No .so produced"
    except subprocess.CalledProcessError as e:
        return False, e.stderr[-800:] if e.stderr else "build failed"
    except subprocess.TimeoutExpired:
        return False, "Extension build timed out"


def run_test_in_subprocess(
    work_dir: Path,
    project_dir: Path,
    ref_path: Path,
    device: int,
) -> dict:
    """在独立子进程中测试（避免 TORCH_LIBRARY 冲突）。"""
    script = textwrap.dedent(f"""\
    import sys, os, json, time, importlib.util
    os.environ["ASCEND_CUSTOM_OPP_PATH"] = "{project_dir / 'build_out'}"
    import torch, torch_npu
    torch.npu.set_device({device})
    device = "npu:{device}"
    result = {{"correctness": None, "max_diff": None, "custom_ms": None, "ref_ms": None, "error": ""}}
    try:
        sys.path.insert(0, "{project_dir / 'ext'}")
        ext = __import__("custom_ext")
        funcs = [n for n in dir(ext) if not n.startswith('_') and callable(getattr(ext, n))]
        if not funcs:
            result["error"] = "No callable in extension"
            print(json.dumps(result)); sys.exit(0)
        ext_func = getattr(ext, funcs[0])

        spec = importlib.util.spec_from_file_location("ref", "{ref_path}")
        ref_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ref_mod)

        init_inputs = getattr(ref_mod, 'get_init_inputs', lambda: [])()
        model = ref_mod.Model(*init_inputs).to(device).eval()
        inputs = [x.to(device) if isinstance(x, torch.Tensor) else x for x in ref_mod.get_inputs()]

        with torch.no_grad():
            y_ref = model(*inputs)
        tensor_inputs = [x for x in inputs if isinstance(x, torch.Tensor)]
        with torch.no_grad():
            y_out = ext_func(*tensor_inputs)

        if isinstance(y_ref, torch.Tensor) and isinstance(y_out, torch.Tensor):
            diff = torch.max(torch.abs(y_ref.float() - y_out.float())).item()
            result["max_diff"] = diff
            result["correctness"] = diff < 1e-2

        torch.npu.synchronize()
        for _ in range(3): ext_func(*tensor_inputs)
        torch.npu.synchronize()
        ts = []
        for _ in range(10):
            torch.npu.synchronize(); t0 = time.perf_counter()
            ext_func(*tensor_inputs)
            torch.npu.synchronize(); ts.append((time.perf_counter()-t0)*1000)
        result["custom_ms"] = sum(ts)/len(ts)

        for _ in range(3): model(*inputs)
        torch.npu.synchronize()
        ts2 = []
        for _ in range(10):
            torch.npu.synchronize(); t0 = time.perf_counter()
            model(*inputs)
            torch.npu.synchronize(); ts2.append((time.perf_counter()-t0)*1000)
        result["ref_ms"] = sum(ts2)/len(ts2)
    except Exception as e:
        result["error"] = f"{{type(e).__name__}}: {{str(e)[:300]}}"
    print(json.dumps(result))
    """)

    try:
        proc = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True, text=True, timeout=120,
            env={**os.environ, "ASCEND_DEVICE_ID": str(device)},
        )
        lines = [l for l in proc.stdout.strip().splitlines() if l.strip()]
        if lines:
            return json.loads(lines[-1])
        return {"correctness": None, "error": f"No output. stderr: {proc.stderr[-300:]}"}
    except Exception as e:
        return {"correctness": None, "error": str(e)[:300]}


# ─── 主循环 ──────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="EvoKernel-lite: LLM 自动生成 Ascend C kernel")
    parser.add_argument("--ref", required=True, help="PyTorch reference .py 文件路径")
    parser.add_argument("--device", type=int, default=4, help="NPU 设备号")
    parser.add_argument("--max-attempts", type=int, default=5, help="最大尝试轮数")
    parser.add_argument("--work-dir", type=str, default=None, help="工作目录")
    parser.add_argument("--model", type=str, default=None, help="LLM 模型名")
    args = parser.parse_args()

    ref_path = Path(args.ref).resolve()
    ref_code = ref_path.read_text()
    ref_name = ref_path.stem  # e.g. "22_Tanh"

    work_dir = Path(args.work_dir) if args.work_dir else Path(f"/workspace/dev/kernel_gen_demo/{ref_name}")
    work_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== EvoKernel-lite ===")
    print(f"Reference: {ref_path}")
    print(f"Work dir:  {work_dir}")
    print(f"Device:    npu:{args.device}")
    print(f"Max attempts: {args.max_attempts}")
    print()

    client = create_client()
    system_prompt = build_system_prompt()

    attempts: list[Attempt] = []
    history_msgs: list[str] = []

    for round_num in range(1, args.max_attempts + 1):
        attempt = Attempt(round=round_num)
        print(f"--- Round {round_num}/{args.max_attempts} ---")

        # 构建 user message
        if round_num == 1:
            user_msg = f"Generate a complete Ascend C custom operator for this PyTorch reference:\n\n```python\n{ref_code}\n```"
        else:
            # 带上之前的错误信息
            prev = attempts[-1]
            feedback_parts = []
            if not prev.compiled:
                feedback_parts.append(f"Compilation FAILED:\n{prev.compile_error[-1500:]}")
            elif not prev.extension_built:
                feedback_parts.append(f"Extension build FAILED:\n{prev.extension_error[-1000:]}")
            elif prev.correctness is False:
                feedback_parts.append(f"Correctness FAILED: max_diff={prev.max_diff}")
            elif prev.error:
                feedback_parts.append(f"Runtime error: {prev.error[:1000]}")
            feedback = "\n".join(feedback_parts)

            user_msg = (
                f"Previous attempt failed. Here is the error:\n\n{feedback}\n\n"
                f"Please fix the code and generate all files again.\n\n"
                f"Original PyTorch reference:\n```python\n{ref_code}\n```"
            )

        # 调用 LLM
        print("  Calling LLM...", end=" ", flush=True)
        t0 = time.time()
        raw_output = call_llm(client, system_prompt, user_msg, model=args.model)
        print(f"({time.time()-t0:.1f}s)")

        # 解析生成的代码
        codes = parse_generated_code(raw_output)
        missing = [k for k, v in codes.items() if not v]
        if missing:
            attempt.error = f"LLM output missing: {missing}"
            print(f"  ERROR: {attempt.error}")
            attempts.append(attempt)
            continue

        attempt.kernel_code = codes["kernel"]
        attempt.host_code = codes["host"]
        attempt.tiling_code = codes["tiling"]

        op_name = infer_op_name(codes["op_json"])
        print(f"  Op: {op_name}")

        # 保存生成的代码
        round_dir = work_dir / f"round_{round_num}"
        round_dir.mkdir(exist_ok=True)
        for key, content in codes.items():
            suffix = {"op_json": ".json", "tiling": ".h", "host": "_host.cpp",
                      "kernel": "_kernel.cpp", "op_cpp": "_ext.cpp"}
            (round_dir / f"{key}{suffix.get(key, '.txt')}").write_text(content)
        (round_dir / "raw_llm_output.txt").write_text(raw_output)

        # 编译
        print("  Compiling...", end=" ", flush=True)
        project_dir = setup_project(work_dir, codes, op_name)
        comp_ok, comp_err = compile_project(project_dir)
        attempt.compiled = comp_ok
        attempt.compile_error = comp_err
        if not comp_ok:
            print(f"FAILED")
            print(f"    {comp_err[:200]}")
            attempts.append(attempt)
            continue
        print("OK")

        # 构建 Extension
        print("  Building extension...", end=" ", flush=True)
        ext_ok, ext_err = build_extension(work_dir, project_dir, codes)
        attempt.extension_built = ext_ok
        attempt.extension_error = ext_err
        if not ext_ok:
            print(f"FAILED")
            print(f"    {ext_err[:200]}")
            attempts.append(attempt)
            continue
        print("OK")

        # 测试
        print("  Testing...", end=" ", flush=True)
        test_res = run_test_in_subprocess(work_dir, project_dir, ref_path, args.device)
        attempt.correctness = test_res.get("correctness")
        attempt.max_diff = test_res.get("max_diff")
        attempt.latency_ms = test_res.get("custom_ms")
        attempt.ref_latency_ms = test_res.get("ref_ms")
        attempt.error = test_res.get("error", "")

        if attempt.latency_ms and attempt.ref_latency_ms and attempt.latency_ms > 0:
            attempt.speedup = attempt.ref_latency_ms / attempt.latency_ms

        if attempt.correctness:
            print(f"PASS! diff={attempt.max_diff:.2e} custom={attempt.latency_ms:.1f}ms ref={attempt.ref_latency_ms:.1f}ms speedup={attempt.speedup:.2f}x")
            attempts.append(attempt)
            break
        else:
            err_info = attempt.error or f"diff={attempt.max_diff}"
            print(f"FAIL: {err_info[:100]}")
            attempts.append(attempt)
            continue

    # 汇总
    print(f"\n{'='*60}")
    print(f"总结: {len(attempts)} 轮尝试")
    for a in attempts:
        status = "PASS" if a.correctness else ("COMPILE_FAIL" if not a.compiled else ("EXT_FAIL" if not a.extension_built else "FAIL"))
        perf = f" {a.latency_ms:.1f}ms ({a.speedup:.2f}x)" if a.speedup else ""
        print(f"  Round {a.round}: {status}{perf}")

    success = any(a.correctness for a in attempts)
    print(f"\n结果: {'SUCCESS' if success else 'FAILED'}")

    # 保存结果
    summary = {
        "ref": str(ref_path),
        "success": success,
        "total_rounds": len(attempts),
        "attempts": [
            {
                "round": a.round, "compiled": a.compiled, "extension_built": a.extension_built,
                "correctness": a.correctness, "max_diff": a.max_diff,
                "custom_ms": a.latency_ms, "ref_ms": a.ref_latency_ms, "speedup": a.speedup,
                "error": a.error,
            }
            for a in attempts
        ],
    }
    (work_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"详情保存至: {work_dir}/summary.json")


if __name__ == "__main__":
    main()
