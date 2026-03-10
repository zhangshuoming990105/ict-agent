#!/usr/bin/env python3
"""Run UniOpBench artifact tests and build a performance report."""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


REFERENCE_MS_RE = re.compile(r"Reference:\s*([0-9.eE+-]+)\s*ms")
TEST_MS_RE = re.compile(r"Test:\s*([0-9.eE+-]+)\s*ms")
SPEEDUP_RE = re.compile(r"Speedup:\s*([0-9.eE+-]+)x")
SHAPE_RE = re.compile(r"Shape:\s*(\([^)]+\))")
STATUS_RE = re.compile(r"STATUS:\s*(PASSED|FAILED|SKIPPED)")


@dataclass
class TensorInfo:
    name: str
    role: str
    dtype: Any
    shape: Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run UniOpBench artifact/test.py files and generate a performance report."
    )
    parser.add_argument(
        "run_root",
        nargs="?",
        default=None,
        help="UniOpBench run directory. Defaults to the latest run under ./uniopbench/*/runs/*, then falls back to ./task/task_results/uniopbench/*/runs/*.",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable used to run each artifact test.",
    )
    parser.add_argument(
        "--backend",
        choices=["cuda", "triton"],
        default="cuda",
        help="Backend passed to test.py.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=1800,
        help="Per-test timeout in seconds.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only run the first N discovered artifacts.",
    )
    parser.add_argument(
        "--match",
        default=None,
        help="Only run artifact paths containing this substring.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for generated report files. Defaults to <run_root>/perf_reports/.",
    )
    parser.add_argument(
        "--extra-pythonpath",
        action="append",
        default=[],
        help="Additional PYTHONPATH entries prepended when running test.py.",
    )
    return parser.parse_args()


def find_latest_run(repo_root: Path) -> Path:
    patterns = [
        "uniopbench/*/runs/*",
        "task/task_results/uniopbench/*/runs/*",
    ]
    for pattern in patterns:
        run_dirs = sorted(repo_root.glob(pattern))
        if run_dirs:
            return run_dirs[-1]
    raise FileNotFoundError(
        "No run directories found under ./uniopbench/*/runs/* or ./task/task_results/uniopbench/*/runs/*"
    )


def safe_literal_eval(node: ast.AST, context: dict[str, Any]) -> Any:
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Tuple):
        return tuple(safe_literal_eval(elt, context) for elt in node.elts)
    if isinstance(node, ast.List):
        return [safe_literal_eval(elt, context) for elt in node.elts]
    if isinstance(node, ast.Dict):
        return {
            safe_literal_eval(key, context): safe_literal_eval(value, context)
            for key, value in zip(node.keys, node.values)
        }
    if isinstance(node, ast.Name):
        return context.get(node.id, node.id)
    if isinstance(node, ast.Attribute):
        if isinstance(node.value, ast.Name) and node.value.id == "torch":
            return node.attr
        base = safe_literal_eval(node.value, context)
        return f"{base}.{node.attr}"
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        value = safe_literal_eval(node.operand, context)
        if isinstance(value, (int, float)):
            return -value
    try:
        return ast.literal_eval(node)
    except Exception:
        return ast.unparse(node)


def collect_param_defaults(tree: ast.Module) -> dict[str, dict[str, Any]]:
    classes: dict[str, dict[str, Any]] = {}
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        if not node.name.endswith("Params"):
            continue
        defaults: dict[str, Any] = {}
        for stmt in node.body:
            if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name) and stmt.value:
                defaults[stmt.target.id] = safe_literal_eval(stmt.value, defaults)
            elif isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
                defaults[stmt.targets[0].id] = safe_literal_eval(stmt.value, defaults)
        classes[node.name] = defaults
    return classes


def infer_params_class(tree: ast.Module, class_defaults: dict[str, dict[str, Any]]) -> str | None:
    if len(class_defaults) == 1:
        return next(iter(class_defaults))

    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name) or target.id != "params":
            continue
        if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name):
            if node.value.func.id in class_defaults:
                return node.value.func.id

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Name) or node.func.id != "create_suite":
            continue
        for kw in node.keywords:
            if kw.arg == "params_class" and isinstance(kw.value, ast.Name):
                if kw.value.id in class_defaults:
                    return kw.value.id
    return None


def extract_tensor_specs(tree: ast.Module, params: dict[str, Any]) -> list[TensorInfo]:
    tensor_specs: list[TensorInfo] = []
    testcase_call: ast.Call | None = None

    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(target, ast.Name) and target.id == "testcase" for target in node.targets):
            continue
        if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name) and node.value.func.id == "TestCase":
            testcase_call = node.value
            break

    if testcase_call is None:
        return tensor_specs

    tensor_specs_node: ast.AST | None = None
    for kw in testcase_call.keywords:
        if kw.arg == "tensor_specs":
            tensor_specs_node = kw.value
            break

    if not isinstance(tensor_specs_node, (ast.List, ast.Tuple)):
        return tensor_specs

    for elt in tensor_specs_node.elts:
        if not isinstance(elt, ast.Call):
            continue
        if not isinstance(elt.func, ast.Name) or elt.func.id != "TensorSpec":
            continue
        args = list(elt.args)
        name = safe_literal_eval(args[0], params) if len(args) > 0 else None
        dtype = safe_literal_eval(args[1], params) if len(args) > 1 else None
        shape = safe_literal_eval(args[2], params) if len(args) > 2 else None
        role = safe_literal_eval(args[3], params) if len(args) > 3 else None

        if isinstance(dtype, str) and dtype in params:
            dtype = params[dtype]
        if isinstance(shape, tuple) and len(shape) == 1 and isinstance(shape[0], tuple):
            shape = shape[0]

        tensor_specs.append(
            TensorInfo(
                name=str(name),
                role=str(role),
                dtype=dtype,
                shape=shape,
            )
        )

    return tensor_specs


def parse_test_metadata(test_path: Path) -> dict[str, Any]:
    source = test_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(test_path))
    class_defaults = collect_param_defaults(tree)
    params_class = infer_params_class(tree, class_defaults)
    params = class_defaults.get(params_class or "", {})
    tensor_specs = extract_tensor_specs(tree, params)
    inputs = [ts for ts in tensor_specs if ts.role == "input"]
    return {
        "params_class": params_class,
        "params": params,
        "inputs": [
            {
                "name": item.name,
                "dtype": item.dtype,
                "shape": item.shape,
            }
            for item in inputs
        ],
    }


def resolve_with_params(value: Any, params: dict[str, Any]) -> Any:
    if isinstance(value, str) and value in params:
        return resolve_with_params(params[value], params)
    if isinstance(value, tuple):
        resolved = tuple(resolve_with_params(item, params) for item in value)
        if len(resolved) == 1 and isinstance(resolved[0], tuple):
            return resolved[0]
        return resolved
    if isinstance(value, list):
        return [resolve_with_params(item, params) for item in value]
    return value


def normalize_shape(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, tuple):
        return str(value)
    if isinstance(value, list):
        return str(tuple(value))
    return str(value)


def parse_perf_output(output: str) -> dict[str, Any]:
    ref_match = REFERENCE_MS_RE.search(output)
    test_match = TEST_MS_RE.search(output)
    speedup_match = SPEEDUP_RE.search(output)
    status_match = STATUS_RE.search(output)
    shapes = SHAPE_RE.findall(output)
    return {
        "reference_ms": float(ref_match.group(1)) if ref_match else None,
        "test_ms": float(test_match.group(1)) if test_match else None,
        "speedup": float(speedup_match.group(1)) if speedup_match else None,
        "reported_shape": shapes[-1] if shapes else None,
        "reported_status": status_match.group(1).lower() if status_match else None,
    }


def summarize_error(output: str, max_lines: int = 8) -> str:
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    if not lines:
        return ""
    return "\n".join(lines[-max_lines:])


def run_artifact_test(
    artifact_dir: Path,
    python_exec: str,
    backend: str,
    timeout: int,
    extra_pythonpath: list[str],
) -> dict[str, Any]:
    test_path = artifact_dir / "test.py"
    metadata = parse_test_metadata(test_path)
    params = metadata.get("params", {})
    for item in metadata.get("inputs", []):
        item["dtype"] = resolve_with_params(item.get("dtype"), params)
        item["shape"] = resolve_with_params(item.get("shape"), params)

    env = os.environ.copy()
    pythonpath_entries = [str(path) for path in extra_pythonpath if path]
    if env.get("PYTHONPATH"):
        pythonpath_entries.append(env["PYTHONPATH"])
    if pythonpath_entries:
        env["PYTHONPATH"] = os.pathsep.join(pythonpath_entries)

    cmd = [python_exec, "test.py", "--backend", backend]
    started_at = time.time()
    try:
        proc = subprocess.run(
            cmd,
            cwd=artifact_dir,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
        )
        timed_out = False
    except subprocess.TimeoutExpired as exc:
        proc = exc
        timed_out = True

    elapsed = time.time() - started_at

    if timed_out:
        stdout = proc.stdout or ""
        stderr = proc.stderr or ""
        exit_code = None
    else:
        stdout = proc.stdout
        stderr = proc.stderr
        exit_code = proc.returncode

    combined_output = stdout if not stderr else f"{stdout}\n{stderr}"
    perf = parse_perf_output(combined_output)
    operator_key = artifact_dir.parent.name

    return {
        "operator_key": operator_key,
        "artifact_dir": str(artifact_dir),
        "command": cmd,
        "exit_code": exit_code,
        "timed_out": timed_out,
        "elapsed_s": round(elapsed, 3),
        "metadata": metadata,
        "performance": perf,
        "stdout": stdout,
        "stderr": stderr,
        "error_summary": summarize_error(combined_output),
        "ok": (not timed_out) and exit_code == 0 and perf["test_ms"] is not None,
    }


def format_inputs(metadata: dict[str, Any]) -> tuple[str, str]:
    inputs = metadata.get("inputs", [])
    unique_dtypes: list[str] = []
    for item in inputs:
        dtype = str(item.get("dtype"))
        if dtype and dtype not in unique_dtypes:
            unique_dtypes.append(dtype)
    dtypes = ", ".join(unique_dtypes) if unique_dtypes else ""
    shapes = ", ".join(normalize_shape(item.get("shape")) for item in inputs) if inputs else ""
    return dtypes, shapes


def build_markdown(
    results: list[dict[str, Any]],
    run_root: Path,
    python_exec: str,
    backend: str,
) -> str:
    success_count = sum(1 for item in results if item["ok"])
    failure_count = len(results) - success_count
    successful_with_speedup = [
        item for item in results if item["ok"] and item["performance"]["speedup"] is not None
    ]
    slower_kernels = [
        item for item in successful_with_speedup if item["performance"]["speedup"] < 1.0
    ]
    arithmetic_mean_speedup = (
        sum(item["performance"]["speedup"] for item in successful_with_speedup)
        / len(successful_with_speedup)
        if successful_with_speedup
        else None
    )
    lines = [
        "# UniOpBench Performance Report",
        "",
        f"- Run root: `{run_root}`",
        f"- Python: `{python_exec}`",
        f"- Backend: `{backend}`",
        f"- Generated at: `{datetime.now().isoformat(timespec='seconds')}`",
        f"- Total artifacts: `{len(results)}`",
        f"- Successful perf runs: `{success_count}`",
        f"- Failed runs: `{failure_count}`",
        (
            f"- Arithmetic mean speedup: `{arithmetic_mean_speedup:.2f}x`"
            if arithmetic_mean_speedup is not None
            else "- Arithmetic mean speedup: `N/A`"
        ),
        (
            "- Kernels with speedup < 1: `"
            + ", ".join(
                f"{item['operator_key']} ({item['performance']['speedup']:.2f}x)"
                for item in slower_kernels
            )
            + "`"
            if slower_kernels
            else "- Kernels with speedup < 1: `None`"
        ),
        "",
        "| Operator | Status | Ref (ms) | Test (ms) | Speedup | Input dtypes | Input shapes |",
        "|---|---:|---:|---:|---:|---|---|",
    ]

    for item in results:
        perf = item["performance"]
        dtypes, shapes = format_inputs(item["metadata"])
        status = "ok" if item["ok"] else ("timeout" if item["timed_out"] else ("no_perf" if item["exit_code"] == 0 else f"exit={item['exit_code']}"))
        lines.append(
            "| {operator} | {status} | {ref} | {test} | {speedup} | {dtypes} | {shapes} |".format(
                operator=item["operator_key"],
                status=status,
                ref="" if perf["reference_ms"] is None else f"{perf['reference_ms']:.6f}",
                test="" if perf["test_ms"] is None else f"{perf['test_ms']:.6f}",
                speedup="" if perf["speedup"] is None else f"{perf['speedup']:.2f}x",
                dtypes=dtypes or "-",
                shapes=(perf["reported_shape"] or shapes or "-").replace("|", "\\|"),
            )
        )

    failed = [item for item in results if not item["ok"]]
    if failed:
        lines.extend(["", "## Failures", ""])
        for item in failed:
            lines.append(f"### {item['operator_key']}")
            lines.append("")
            lines.append(f"- Artifact: `{item['artifact_dir']}`")
            lines.append(f"- Exit code: `{item['exit_code']}`")
            lines.append(f"- Timed out: `{item['timed_out']}`")
            if item["error_summary"]:
                lines.append("- Error summary:")
                lines.append("```text")
                lines.append(item["error_summary"])
                lines.append("```")
            lines.append("")

    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    run_root = Path(args.run_root).resolve() if args.run_root else find_latest_run(repo_root)

    # Ensure optest (from UniOpBench submodule) is on PYTHONPATH for artifact tests
    uniopbench_root = repo_root / "benchmarks" / "UniOpBench"
    if uniopbench_root.exists():
        extra_pythonpath = [str(uniopbench_root)] + list(args.extra_pythonpath)
    else:
        extra_pythonpath = list(args.extra_pythonpath)

    artifacts = sorted(path.parent for path in run_root.glob("operators/*/artifact/test.py"))
    if args.match:
        artifacts = [path for path in artifacts if args.match in str(path)]
    if args.limit is not None:
        artifacts = artifacts[: args.limit]

    if not artifacts:
        print(f"No artifact/test.py files found under {run_root}", file=sys.stderr)
        return 1

    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else run_root / "perf_reports"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    for index, artifact_dir in enumerate(artifacts, start=1):
        print(f"[{index}/{len(artifacts)}] Running {artifact_dir.relative_to(run_root)}")
        result = run_artifact_test(
            artifact_dir=artifact_dir,
            python_exec=args.python,
            backend=args.backend,
            timeout=args.timeout,
            extra_pythonpath=extra_pythonpath,
        )
        results.append(result)
        perf = result["performance"]
        if result["ok"]:
            print(
                "  ok: test={test:.6f} ms speedup={speedup:.2f}x".format(
                    test=perf["test_ms"],
                    speedup=perf["speedup"],
                )
            )
        else:
            print(f"  failed: {result['error_summary'].splitlines()[-1] if result['error_summary'] else 'unknown error'}")

    json_path = output_dir / "perf_report.json"
    md_path = output_dir / "perf_report.md"

    report_payload = {
        "run_root": str(run_root),
        "python": args.python,
        "backend": args.backend,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "results": results,
    }
    json_path.write_text(json.dumps(report_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    md_path.write_text(
        build_markdown(results, run_root=run_root, python_exec=args.python, backend=args.backend),
        encoding="utf-8",
    )

    print(f"\nJSON report: {json_path}")
    print(f"Markdown report: {md_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
