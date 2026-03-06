"""GPU helpers for CUDA task execution."""

from __future__ import annotations

import json
import subprocess


def _query_rocm_smi() -> list[dict] | None:
    try:
        proc = subprocess.run(
            ["rocm-smi", "--showuse", "--showmemuse", "--json"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except FileNotFoundError:
        return None
    except Exception:
        return None
    if proc.returncode != 0:
        return None
    try:
        data = json.loads(proc.stdout)
    except (json.JSONDecodeError, TypeError):
        return None
    gpus: list[dict] = []
    for key in sorted(data.keys()):
        if not key.startswith("card"):
            continue
        idx = key[len("card") :]
        info = data[key]
        util = float(info.get("HCU use (%)", info.get("GPU use (%)", "100")))
        mem_pct = float(info.get("HCU memory use (%)", info.get("GPU memory use (%)", "100")))
        gpus.append({"index": idx, "util": util, "mem_pct": mem_pct})
    return gpus if gpus else None


def _query_nvidia_smi() -> list[dict] | None:
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except FileNotFoundError:
        return None
    except Exception:
        return None
    if proc.returncode != 0:
        return None
    gpus: list[dict] = []
    for line in proc.stdout.strip().splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 4:
            continue
        try:
            idx = parts[0]
            util = float(parts[1])
            mem_used = float(parts[2])
            mem_total = float(parts[3])
            mem_pct = (mem_used / mem_total * 100) if mem_total > 0 else 100.0
        except (ValueError, ZeroDivisionError):
            continue
        gpus.append({"index": idx, "util": util, "mem_pct": mem_pct})
    return gpus if gpus else None


def query_gpus() -> list[dict]:
    result = _query_rocm_smi()
    if result is not None:
        return result
    result = _query_nvidia_smi()
    if result is not None:
        return result
    return []


def acquire_gpu() -> str | None:
    for gpu in query_gpus():
        if gpu["util"] == 0.0 and gpu["mem_pct"] < 1.0:
            return gpu["index"]
    return None


def gpu_status_summary() -> str:
    gpus = query_gpus()
    if not gpus:
        return "No GPU status available (rocm-smi/nvidia-smi not found)"
    parts: list[str] = []
    for gpu in gpus:
        status = "idle" if (gpu["util"] == 0.0 and gpu["mem_pct"] < 1.0) else "busy"
        parts.append(
            f"GPU{gpu['index']}:{status}(util={gpu['util']:.0f}%,mem={gpu['mem_pct']:.0f}%)"
        )
    return " ".join(parts)
