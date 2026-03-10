"""CUDA-specific failure classification and recovery hints."""

from __future__ import annotations

from dataclasses import dataclass, field
import re


COMPILE_FAIL_RE = re.compile(r"Compilation failed", re.IGNORECASE)
PROFILE_RESULT_RE = re.compile(
    r"Torch Baseline:\s*([\d.]+)us.*Torch Compile:\s*([\d.]+)us.*CUDA Extension:\s*([\d.]+)us",
    re.IGNORECASE,
)


class CudaFailureKind:
    COMPILE = "compile"
    CORRECTNESS = "correctness"
    PERFORMANCE = "performance"
    GENERAL = "general"


EXIT_CODE_RE = re.compile(r"^exit_code=(-?\d+)$", re.MULTILINE)


@dataclass
class ToolExecutionOutcome:
    called: bool = False
    failures: list[str] = field(default_factory=list)
    failure_kinds: list[str] = field(default_factory=list)


@dataclass
class RecoveryState:
    had_failure: bool = False
    unresolved_failure: bool = False
    repeated_failure_count: int = 0
    last_failure_signature: str = ""
    failures: list[str] = field(default_factory=list)
    last_failure_kind: str = CudaFailureKind.GENERAL

    def record_failures(self, failures: list[str], kinds: list[str] | None = None) -> None:
        if not failures:
            self.unresolved_failure = False
            self.repeated_failure_count = 0
            return
        self.had_failure = True
        self.unresolved_failure = True
        signature = " | ".join(failures)
        if signature == self.last_failure_signature:
            self.repeated_failure_count += 1
        else:
            self.repeated_failure_count = 1
            self.last_failure_signature = signature
        self.failures.extend(failures[-3:])
        if kinds:
            self.last_failure_kind = kinds[-1]


def classify_cuda_failure(tool_name: str, result: str) -> str:
    if tool_name != "run_shell":
        return CudaFailureKind.GENERAL
    text = (result or "").strip()
    if COMPILE_FAIL_RE.search(text) or "error:" in text.lower():
        if any(
            keyword in text.lower()
            for keyword in ("undefined symbol", "nvcc", "hipcc", "syntax error", "compilation failed")
        ):
            return CudaFailureKind.COMPILE
    if "AssertionError" in text or "assert_close" in text or "RuntimeError" in text:
        return CudaFailureKind.CORRECTNESS
    match = PROFILE_RESULT_RE.search(text)
    if match:
        try:
            compile_time = float(match.group(2))
            cuda_time = float(match.group(3))
            if cuda_time > compile_time * 0.95:
                return CudaFailureKind.PERFORMANCE
        except ValueError:
            pass
    return CudaFailureKind.GENERAL


def is_tool_failure(tool_name: str, result: str) -> bool:
    text = (result or "").strip()
    if not text:
        return False
    if text.startswith("Error") or text.startswith("Denied"):
        return True
    if tool_name == "run_shell":
        match = EXIT_CODE_RE.search(text)
        if match:
            try:
                return int(match.group(1)) != 0
            except ValueError:
                return True
    return False


def summarize_failure(tool_name: str, result: str) -> str:
    first_line = (result or "").strip().splitlines()[0] if result else ""
    if not first_line:
        first_line = "unknown failure"
    return f"{tool_name}: {first_line}"


def build_recovery_nudge(recovery: RecoveryState) -> str:
    latest = recovery.failures[-1] if recovery.failures else "unknown failure"
    repeat_hint = ""
    if recovery.repeated_failure_count >= 2:
        repeat_hint = (
            "You repeated a similar failing attempt. Change strategy and do not retry the same "
            "command unchanged.\n"
        )
    kind = recovery.last_failure_kind
    if kind == CudaFailureKind.COMPILE:
        specific = (
            "This is a COMPILATION error. Check extern declarations, include paths, kernel syntax, "
            "and registration macros.\n"
        )
    elif kind == CudaFailureKind.CORRECTNESS:
        specific = (
            "This is a CORRECTNESS error. Check boundary conditions, indexing, synchronization, "
            "and model_new.py parity with the original model.\n"
        )
    elif kind == CudaFailureKind.PERFORMANCE:
        specific = (
            "Correctness passed but PERFORMANCE is insufficient. Prioritize kernel fusion, shared "
            "memory tiling, vectorized loads, and occupancy tuning.\n"
        )
    else:
        specific = ""
    return (
        "Recovery mode: latest tool execution failed.\n"
        f"- Latest failure: {latest}\n"
        f"{specific}"
        "- Diagnose root cause from tool output and perform a concrete fix in the next tool call.\n"
        "- Do not ask user for confirmation; continue autonomously.\n"
        f"{repeat_hint}"
    ).strip()
