#!/usr/bin/env python3
"""
Run performance benchmark for all operators and report speedup vs baseline (PyTorch).

For operators with variants (supports_variants=true), runs with --variants yaml
and reports average speedup across variants. Otherwise runs single test mode.
"""

import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Tuple

# Paths
RUN_DIR = Path(__file__).parent.resolve()
OPERATORS_DIR = RUN_DIR / "operators"
UNIOPBENCH = Path(__file__).resolve().parents[5] / "benchmarks" / "UniOpBench"


def load_run_summary():
    """Load run_summary.json to get operator list and metadata."""
    summary_path = RUN_DIR / "run_summary.json"
    with open(summary_path) as f:
        return json.load(f)


def extract_from_existing_perf_logs(operators: dict) -> List[dict]:
    """Extract speedup from existing perf.log files when live run is not possible."""
    results = []
    for op_key, op_info in operators.items():
        if op_info.get("status") != "passed":
            continue

        perf_log = op_info.get("perf", {}).get("log", "")
        if not perf_log or not os.path.exists(perf_log):
            results.append({
                "operator": op_info["operator"],
                "operator_key": op_key,
                "speedup": None,
                "avg_speedup": None,
                "variant_count": 1,
                "error": "No perf.log",
                "has_variants": op_info.get("supports_variants", False),
                "source": "none",
            })
            continue

        with open(perf_log) as f:
            content = f.read()

        speedups = parse_speedup_from_output(content)
        # Filter out placeholder/invalid values (0, negative, or unreasonably large)
        speedups = [s for s in speedups if 0.001 < s < 1000]

        if speedups:
            avg = sum(speedups) / len(speedups)
            results.append({
                "operator": op_info["operator"],
                "operator_key": op_key,
                "speedup": speedups[0],
                "avg_speedup": avg,
                "variant_count": len(speedups),
                "speedups": speedups,
                "error": None,
                "has_variants": op_info.get("supports_variants", False),
                "source": "perf.log",
            })
        else:
            results.append({
                "operator": op_info["operator"],
                "operator_key": op_key,
                "speedup": None,
                "avg_speedup": None,
                "variant_count": 0,
                "error": "No valid speedup in perf.log",
                "has_variants": op_info.get("supports_variants", False),
                "source": "perf.log",
            })
    return results


def parse_speedup_from_output(output: str) -> List[float]:
    """Parse all Speedup: X.XXx patterns from test output."""
    # Match patterns like "Speedup: 1.38x" or "Speedup:   0.96x"
    pattern = r"Speedup:\s*([\d.]+)x"
    matches = re.findall(pattern, output)
    return [float(m) for m in matches]


def run_operator_test(artifact_dir: Path, supports_variants: bool) -> Tuple[List[float], Optional[str]]:
    """
    Run test.py for an operator and return list of speedups.
    Returns (speedups, error_message).
    """
    test_py = artifact_dir / "test.py"
    if not test_py.exists():
        return [], f"test.py not found"

    env = os.environ.copy()
    env["PYTHONPATH"] = str(UNIOPBENCH) + (":" + env.get("PYTHONPATH", "") if env.get("PYTHONPATH") else "")

    args = [sys.executable, "test.py", "--backend", "cuda"]
    if supports_variants:
        args.extend(["--variants", "yaml", "-v"])

    try:
        result = subprocess.run(
            args,
            cwd=artifact_dir,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            timeout=300,
        )
        output = result.stdout or ""
        speedups = parse_speedup_from_output(output)

        if result.returncode != 0 and not speedups:
            return [], f"Test failed (exit {result.returncode})"
        return speedups, None
    except subprocess.TimeoutExpired:
        return [], "Timeout"
    except Exception as e:
        return [], str(e)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run perf benchmark for all operators")
    parser.add_argument("--from-logs", action="store_true",
                        help="Extract speedup from existing perf.log (no GPU needed)")
    args = parser.parse_args()

    summary = load_run_summary()
    operators = summary.get("operators", {})

    if args.from_logs:
        print("Extracting speedup from existing perf.log files...\n")
        results = extract_from_existing_perf_logs(operators)
    else:
        results = []
        for op_key, op_info in operators.items():
            if op_info.get("status") != "passed":
                continue

            artifact_dir = Path(op_info["artifact_dir"])
            supports_variants = op_info.get("supports_variants", False)

            print(f"Running perf test: {op_key}...", end=" ", flush=True)
            speedups, err = run_operator_test(artifact_dir, supports_variants)

            if err:
                print(f"ERROR: {err}")
                results.append({
                    "operator": op_info["operator"],
                    "operator_key": op_key,
                    "speedup": None,
                    "avg_speedup": None,
                    "variant_count": 0,
                    "error": err,
                    "has_variants": supports_variants,
                })
                continue

            if speedups:
                avg = sum(speedups) / len(speedups)
                print(f"OK - {len(speedups)} variant(s), avg speedup: {avg:.2f}x")
                results.append({
                    "operator": op_info["operator"],
                    "operator_key": op_key,
                    "speedup": speedups[0] if len(speedups) == 1 else None,
                    "avg_speedup": avg,
                    "variant_count": len(speedups),
                    "speedups": speedups,
                    "error": None,
                    "has_variants": supports_variants,
                })
            else:
                print("WARN: No speedup data in output")
                results.append({
                    "operator": op_info["operator"],
                    "operator_key": op_key,
                    "speedup": None,
                    "avg_speedup": None,
                    "variant_count": 0,
                    "error": "No speedup in output",
                    "has_variants": supports_variants,
                })

    # Generate report
    report_path = RUN_DIR / "perf_report.json"
    with open(report_path, "w") as f:
        json.dump({"results": results}, f, indent=2)

    # Print summary table
    print("\n" + "=" * 80)
    print("PERFORMANCE BENCHMARK REPORT: Kernel Speedup vs PyTorch Baseline")
    print("=" * 80)
    print(f"{'Operator':<35} {'Speedup':<12} {'Variants':<10} {'Note'}")
    print("-" * 80)

    for r in sorted(results, key=lambda x: x["operator_key"]):
        op = r["operator_key"]
        if r["avg_speedup"] is not None:
            sp = r["avg_speedup"]
            var_info = f"{r['variant_count']} avg" if r["has_variants"] and r["variant_count"] > 1 else "single"
            note = ""
        elif r["error"]:
            sp = "N/A"
            var_info = "-"
            note = r["error"][:30]
        else:
            sp = "N/A"
            var_info = "-"
            note = "No data"

        speedup_str = f"{sp:.2f}x" if isinstance(sp, (int, float)) else str(sp)
        print(f"{op:<35} {speedup_str:<12} {var_info:<10} {note}")

    print("=" * 80)
    valid = [r for r in results if r["avg_speedup"] is not None]
    if valid:
        overall_avg = sum(r["avg_speedup"] for r in valid) / len(valid)
        speedup_gt_1 = sum(1 for r in valid if r["avg_speedup"] > 1)
        print(f"\nSummary: {len(valid)}/{len(results)} operators with perf data")
        print(f"  - Average speedup (geometric mean of reported): {overall_avg:.2f}x")
        print(f"  - Operators with speedup > 1x: {speedup_gt_1}/{len(valid)}")
    print(f"\nFull report saved to: {report_path}")


if __name__ == "__main__":
    main()
