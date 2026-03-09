"""UniOpBench experiment orchestrator owned by this repository."""

from __future__ import annotations

import json
import os
import re
import select
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import time
from typing import Any

import yaml

_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*m")
_READY_MARKER = ">>> Ready for input."


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def benchmark_root() -> Path:
    return repo_root() / "benchmarks" / "UniOpBench"


def task_config_path() -> Path:
    return repo_root() / "task" / "uniopbench" / "task.yaml"


def task_template_path() -> Path:
    return repo_root() / "task" / "uniopbench" / "TASK.md"


def experiment_dir(experiment_name: str) -> Path:
    """Return experiment directory (task_results/uniopbench/<experiment.name>)."""
    return repo_root() / "task" / "task_results" / "uniopbench" / experiment_name


def runs_root(experiment_name: str) -> Path:
    """Return runs directory for the given experiment. Path includes experiment name from task.yaml."""
    return experiment_dir(experiment_name) / "runs"


def timestamp_run_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def safe_operator_name(operator: str) -> str:
    return operator.replace("/", "__")


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


class Logger:
    def __init__(self, path: Path):
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self._path.open("a", encoding="utf-8")

    def log(self, message: str) -> None:
        print(message, flush=True)
        self._handle.write(message + "\n")
        self._handle.flush()

    def close(self) -> None:
        self._handle.close()


@dataclass
class ExperimentConfig:
    name: str
    model: str
    notes: str = ""
    provider: str = "auto"
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    max_tokens: int = 8192
    max_repair_rounds: int = 3
    max_agent_steps: int | None = 0  # 0 = unlimited; None = use formula max(8, max_repair_rounds*4)
    cuda_arch: str = "sm_80"
    enable_torch_compile_baseline: bool = True


@dataclass
class PromptConfig:
    use_task_template: bool = True
    extra_system: str = ""
    extra_user: str = ""


@dataclass
class RuntimeConfig:
    stop_on_first_failure: bool = False
    keep_temp_builds: bool = True


@dataclass
class TaskConfig:
    experiment: ExperimentConfig
    operators: list[str]
    prompt: PromptConfig = field(default_factory=PromptConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)


def list_all_operators() -> list[str]:
    root = benchmark_root() / "operators"
    operators: list[str] = []
    for test_file in sorted(root.rglob("test.py")):
        operators.append(str(test_file.parent.relative_to(root)))
    return operators


def normalize_operators(operators: list[str]) -> list[str]:
    expanded: list[str] = []
    for operator in operators:
        operator_text = str(operator).strip()
        if not operator_text:
            continue
        if operator_text.lower() == "all":
            expanded.extend(list_all_operators())
            continue
        operator_path = Path(operator_text)
        if operator_path.is_absolute() or ".." in operator_path.parts:
            raise ValueError(f"Invalid operator path: {operator_text}")
        expanded.append(operator_text)

    deduped: list[str] = []
    seen: set[str] = set()
    for operator in expanded:
        if operator in seen:
            continue
        seen.add(operator)
        deduped.append(operator)
    return deduped


def load_task_config(config_path: Path, operators_override: list[str] | None = None) -> TaskConfig:
    if not config_path.is_file():
        raise FileNotFoundError(f"Task config not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    experiment_raw = raw.get("experiment") or {}
    operators = operators_override or raw.get("operators") or []
    if not experiment_raw.get("model"):
        raise ValueError("task.yaml is missing experiment.model")
    if not operators:
        raise ValueError("task.yaml must define at least one operator")

    normalized_operators = normalize_operators(operators)
    if not normalized_operators:
        raise ValueError("No valid operators resolved from task.yaml")

    max_repair_rounds = int(experiment_raw.get("max_repair_rounds", 3))
    if max_repair_rounds < 0:
        raise ValueError("experiment.max_repair_rounds must be >= 0")

    max_agent_steps_raw = experiment_raw.get("max_agent_steps")
    max_agent_steps: int | None = 0  # default: unlimited
    if max_agent_steps_raw is not None:
        max_agent_steps = int(max_agent_steps_raw)
        if max_agent_steps < 0:
            raise ValueError("experiment.max_agent_steps must be >= 0")

    return TaskConfig(
        experiment=ExperimentConfig(
            name=str(experiment_raw.get("name") or "uniopbench_run"),
            notes=str(experiment_raw.get("notes") or ""),
            provider=str(experiment_raw.get("provider") or "auto"),
            model=str(experiment_raw["model"]),
            temperature=float(t) if (t := experiment_raw.get("temperature")) is not None else None,
            top_p=float(p) if (p := experiment_raw.get("top_p")) is not None else None,
            top_k=int(k) if (k := experiment_raw.get("top_k")) is not None else None,
            max_tokens=int(experiment_raw.get("max_tokens", 32768)),
            max_repair_rounds=max_repair_rounds,
            max_agent_steps=max_agent_steps,
            cuda_arch=str(experiment_raw.get("cuda_arch") or "sm_80"),
            enable_torch_compile_baseline=bool(
                experiment_raw.get("enable_torch_compile_baseline", True)
            ),
        ),
        operators=normalized_operators,
        prompt=PromptConfig(**(raw.get("prompt") or {})),
        runtime=RuntimeConfig(**(raw.get("runtime") or {})),
    )


def operator_source_dir(operator: str) -> Path:
    op_dir = benchmark_root() / "operators" / operator
    required = [
        op_dir / "test.py",
        op_dir / "cases.yaml",
        op_dir / "torch_" / "ref.py",
        op_dir / "cuda_" / "kernel.cu",
    ]
    if not all(path.exists() for path in required):
        raise FileNotFoundError(f"Invalid UniOpBench operator: {operator}")
    return op_dir


def copy_operator_tree(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(
        src,
        dst,
        ignore=shutil.ignore_patterns(
            "__pycache__",
            "*.pyc",
            "*.pyo",
            "*.so",
            "lib_cuda_kernel.so",
            ".DS_Store",
            "kernel.cu",
            "kernel.py"
        ),
    )


def supports_variants(test_file: Path) -> bool:
    return "--variants" in test_file.read_text(encoding="utf-8", errors="replace")


def prompt_file_list(operator_dir: Path) -> list[Path]:
    paths = [
        operator_dir / "test.py",
        operator_dir / "torch_" / "ref.py",
        operator_dir / "cases.yaml",
        operator_dir / "get_data.py",
        operator_dir / "check_cuda.py",
        operator_dir / "check_triton.py",
    ]
    return [path for path in paths if path.exists()]


def build_prompt(
    task_config: TaskConfig,
    operator: str,
    operator_dir: Path,
    prior_round: dict[str, Any] | None,
    artifact_rel_path: str = "",
) -> tuple[str, str]:
    system_parts = [
        "You are operating as the full ict-agent runtime inside an isolated UniOpBench operator workspace.",
        "Use tools to inspect files, edit cuda_/kernel.cu, and run the existing test.py commands autonomously.",
        "Do not modify any file other than cuda_/kernel.cu unless the scaffold itself is broken.",
        "The generated kernel must preserve the existing exported C interface expected by the operator's test.py.",
        "Turn-specific request files may appear under .uniopbench/requests/; read the file referenced by the user turn and treat it as the detailed task instruction for that turn.",
    ]
    if artifact_rel_path:
        system_parts.append(
            f"The operator artifact directory is: {artifact_rel_path}\n"
            f"- Kernel file: {artifact_rel_path}/cuda_/kernel.cu\n"
            f"- Always use run_shell with cwd=\"{artifact_rel_path}\" when running test.py commands."
        )
    else:
        system_parts.append(
            "The workspace root is the operator artifact directory. Run test.py commands directly from the workspace."
        )
    if task_config.prompt.use_task_template and task_template_path().is_file():
        system_parts.append(task_template_path().read_text(encoding="utf-8"))
    if task_config.prompt.extra_system:
        system_parts.append(task_config.prompt.extra_system)

    cmd_hint = (
        f' (run from cwd="{artifact_rel_path}" if workspace is experiment root)'
        if artifact_rel_path
        else ""
    )
    user_parts = [
        f"Operator: {operator}",
        f"Target GPU architecture: {task_config.experiment.cuda_arch}",
        "Read the workspace files, update cuda_/kernel.cu, and validate your changes with:",
        f"python test.py --compile-only{cmd_hint}",
        f"python test.py --no-perf{cmd_hint}",
        f"python test.py{cmd_hint}",
        f"If variants are supported: python test.py --variants yaml --no-perf{cmd_hint}",
        "Work autonomously in this turn. When compile and correctness pass, reply with a short summary.",
    ]
    for path in prompt_file_list(operator_dir):
        rel = path.relative_to(operator_dir)
        lang = path.suffix.lstrip(".") or "text"
        user_parts.append(
            f"\n## File: {rel}\n```{lang}\n{path.read_text(encoding='utf-8', errors='replace')}\n```"
        )

    if prior_round is not None:
        user_parts.append("\n## Previous kernel\n```cuda\n" + prior_round["kernel"] + "\n```")
        if prior_round.get("compile_log"):
            user_parts.append("\n## Compile log\n```text\n" + prior_round["compile_log"] + "\n```")
        if prior_round.get("verify_log"):
            user_parts.append("\n## Verification log\n```text\n" + prior_round["verify_log"] + "\n```")
        if prior_round.get("perf_log"):
            user_parts.append("\n## Performance log\n```text\n" + prior_round["perf_log"] + "\n```")
        if prior_round.get("variants_log"):
            user_parts.append("\n## Variants log\n```text\n" + prior_round["variants_log"] + "\n```")
        user_parts.append(
            "\nRevise the kernel to fix the reported failures while keeping the same interface."
        )
    if task_config.prompt.extra_user:
        user_parts.append(task_config.prompt.extra_user)

    return "\n\n".join(system_parts), "\n".join(user_parts)


def _resolve_max_agent_steps(experiment: ExperimentConfig) -> int:
    """Resolve max_agent_steps: None = formula, 0 = unlimited."""
    if experiment.max_agent_steps is not None:
        return experiment.max_agent_steps
    return max(8, experiment.max_repair_rounds * 4)


@dataclass
class UsageSnapshot:
    requests: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


def _parse_usage_snapshot(lines: list[str]) -> UsageSnapshot:
    snapshot = UsageSnapshot()
    patterns = {
        "requests": r"^Requests:\s+([0-9,]+)$",
        "prompt_tokens": r"^\s*Prompt:\s+([0-9,]+)\s+tokens$",
        "completion_tokens": r"^\s*Completion:\s+([0-9,]+)\s+tokens$",
        "total_tokens": r"^\s*Total:\s+([0-9,]+)\s+tokens$",
    }
    for raw_line in lines:
        line = raw_line.strip()
        for field_name, pattern in patterns.items():
            match = re.match(pattern, line)
            if match:
                setattr(snapshot, field_name, int(match.group(1).replace(",", "")))
    return snapshot


def _usage_delta(current: UsageSnapshot, previous: UsageSnapshot) -> dict[str, int]:
    return {
        "prompt_tokens": max(0, current.prompt_tokens - previous.prompt_tokens),
        "completion_tokens": max(0, current.completion_tokens - previous.completion_tokens),
        "total_tokens": max(0, current.total_tokens - previous.total_tokens),
    }


def _parse_ctx_messages(lines: list[str]) -> list[dict[str, Any]] | None:
    start_idx: int | None = None
    for idx, raw_line in enumerate(lines):
        if raw_line.strip() == "[":
            start_idx = idx
            break
    if start_idx is None:
        return None

    json_lines: list[str] = []
    for raw_line in lines[start_idx:]:
        stripped = raw_line.strip()
        if stripped == _READY_MARKER:
            break
        if raw_line.lstrip().startswith("//"):
            continue
        json_lines.append(raw_line.rstrip("\n"))

    raw = "\n".join(json_lines).strip()
    if not raw:
        return None
    end_idx = raw.rfind("]")
    if end_idx < 0:
        return None
    try:
        payload = json.loads(raw[: end_idx + 1])
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, list) else None


def _parse_turn_result(lines: list[str]) -> tuple[str, str, bool, bool, str]:
    assistant_content = ""
    response_model = ""
    tool_called = any("-> Calling tool:" in line for line in lines)
    had_failure = any("[recovery] Detected failure" in line for line in lines)
    error = ""
    last_logged_error = ""

    idx = 0
    while idx < len(lines):
        line = lines[idx].rstrip("\n")
        if line.startswith("Assistant ["):
            match = re.match(r"Assistant \[(.+?)\]:\s?(.*)$", line)
            if match:
                response_model = match.group(1)
                content_lines = [match.group(2)] if match.group(2) else []
                idx += 1
                while idx < len(lines):
                    next_line = lines[idx].rstrip("\n")
                    if next_line.startswith("  [tokens:") or next_line.strip() == _READY_MARKER:
                        break
                    content_lines.append(next_line)
                    idx += 1
                assistant_content = "\n".join(content_lines).strip()
                continue
        if line.startswith("Error: "):
            last_logged_error = line[len("Error: ") :].strip()
        idx += 1

    if "I reached the max autonomous step limit" in assistant_content:
        error = "max_agent_steps_reached"
    elif last_logged_error:
        error = last_logged_error

    return assistant_content, response_model, tool_called, had_failure, error


def _snapshot_agent_log(session_log_path: Path, round_dir: Path) -> None:
    if not session_log_path.is_file():
        return
    shutil.copy2(session_log_path, round_dir / "agent.log")


class LiveAgentSession:
    def __init__(
        self,
        task_config: TaskConfig,
        workspace_dir: Path,
        system_prompt_path: Path,
        session_log_path: Path,
    ) -> None:
        runner_path = repo_root() / "task" / "uniopbench" / "session_runner.py"
        cmd = [
            sys.executable,
            str(runner_path),
            "--workspace",
            str(workspace_dir),
            "--system-prompt-file",
            str(system_prompt_path),
            "--log-path",
            str(session_log_path),
            "--provider",
            task_config.experiment.provider,
            "--model",
            task_config.experiment.model,
            "--max-tokens",
            str(128_000),
            "--max-agent-steps",
            str(_resolve_max_agent_steps(task_config.experiment)),
            "--compact-model",
            task_config.experiment.model,
        ]
        self._proc = subprocess.Popen(
            cmd,
            cwd=repo_root(),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        self._session_log_path = session_log_path
        self._usage_snapshot = UsageSnapshot()
        self._last_output_tail: list[str] = []
        self._read_until_ready(timeout_sec=120)

    @property
    def session_log_path(self) -> Path:
        return self._session_log_path

    def _remember_lines(self, lines: list[str]) -> None:
        self._last_output_tail.extend(lines)
        if len(self._last_output_tail) > 200:
            self._last_output_tail = self._last_output_tail[-200:]

    def _read_until_ready(self, timeout_sec: int) -> list[str]:
        if self._proc.stdout is None:
            raise RuntimeError("Live agent session stdout is unavailable.")
        deadline = time.monotonic() + timeout_sec
        lines: list[str] = []
        while time.monotonic() < deadline:
            ready, _, _ = select.select([self._proc.stdout], [], [], 1.0)
            if not ready:
                if self._proc.poll() is not None:
                    break
                continue
            line = self._proc.stdout.readline()
            if line == "":
                if self._proc.poll() is not None:
                    break
                continue
            lines.append(line)
            if line.strip() == _READY_MARKER:
                self._remember_lines(lines)
                return lines
        self._remember_lines(lines)
        tail = "".join(self._last_output_tail[-40:])
        raise TimeoutError(
            "Timed out waiting for live agent readiness."
            + (f"\nLast output:\n{tail}" if tail else "")
        )

    def send(self, message: str, timeout_sec: int = 1800) -> list[str]:
        if "\n" in message:
            raise ValueError("Live agent session only accepts single-line user inputs.")
        if self._proc.stdin is None:
            raise RuntimeError("Live agent session stdin is unavailable.")
        if self._proc.poll() is not None:
            raise RuntimeError(f"Live agent session exited with code {self._proc.returncode}.")
        self._proc.stdin.write(message.rstrip("\n") + "\n")
        self._proc.stdin.flush()
        return self._read_until_ready(timeout_sec=timeout_sec)

    def run_round(
        self,
        round_dir: Path,
        request_payload: dict[str, Any],
        live_user_input: str,
    ) -> dict[str, Any]:
        turn_output = self.send(live_user_input, timeout_sec=1800)
        token_output = self.send("/tokens", timeout_sec=60)
        raw_output = self.send("/debug raw", timeout_sec=60)

        assistant_content, response_model, tool_called, had_failure, error = _parse_turn_result(
            turn_output
        )
        usage_snapshot = _parse_usage_snapshot(token_output)
        token_usage = _usage_delta(usage_snapshot, self._usage_snapshot)
        steps = max(0, usage_snapshot.requests - self._usage_snapshot.requests)
        self._usage_snapshot = usage_snapshot
        ctx_messages = _parse_ctx_messages(raw_output)

        response_payload = {
            "assistant_content": assistant_content,
            "response_model": response_model or request_payload["model"],
            "steps": steps,
            "tool_called": tool_called,
            "had_failure": had_failure,
            "error": error,
            "ctx_messages": ctx_messages,
            "token_usage": token_usage,
            "cumulative_usage": {
                "requests": usage_snapshot.requests,
                "prompt_tokens": usage_snapshot.prompt_tokens,
                "completion_tokens": usage_snapshot.completion_tokens,
                "total_tokens": usage_snapshot.total_tokens,
            },
            "live_user_input": live_user_input,
        }

        if ctx_messages:
            from ict_agent.context import ContextManager

            ctx = ContextManager(system_prompt="", max_tokens=128_000)
            ctx.messages = ctx_messages
            plain = _ANSI_ESCAPE_RE.sub("", ctx.format_debug())
            write_text(round_dir / "trajectory.log", plain)

        _snapshot_agent_log(self._session_log_path, round_dir)
        return response_payload

    def close(self) -> None:
        if self._proc.poll() is not None:
            return
        try:
            if self._proc.stdin is not None:
                self._proc.stdin.write("quit\n")
                self._proc.stdin.flush()
            self._proc.wait(timeout=10)
        except Exception:
            self._proc.kill()
            self._proc.wait(timeout=5)


def subprocess_env(task_config: TaskConfig) -> dict[str, str]:
    env = os.environ.copy()
    python_path_entries = [str(benchmark_root()), env.get("PYTHONPATH", "")]
    env["PYTHONPATH"] = os.pathsep.join([entry for entry in python_path_entries if entry])
    env["UNIOPBENCH_TASK_CUDA_ARCH"] = task_config.experiment.cuda_arch
    env["UNIOPBENCH_TASK_COMPILE_BASELINE"] = (
        "1" if task_config.experiment.enable_torch_compile_baseline else "0"
    )
    return env


def run_command(
    command: list[str],
    cwd: Path,
    env: dict[str, str],
    log_path: Path,
) -> tuple[int, str]:
    proc = subprocess.run(
        command,
        cwd=cwd,
        env=env,
        text=True,
        capture_output=True,
    )
    output = (proc.stdout or "") + (proc.stderr or "")
    write_text(log_path, output)
    return proc.returncode, output


def load_existing_summary(run_dir: Path) -> dict[str, Any]:
    summary_path = run_dir / "run_summary.json"
    if not summary_path.is_file():
        return {"operators": {}}
    with summary_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def cleanup_artifact_tree(artifact_dir: Path) -> None:
    for path in artifact_dir.rglob("__pycache__"):
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
    for pattern in ("*.pyc", "*.pyo", "*.so"):
        for path in artifact_dir.rglob(pattern):
            if path.is_file():
                path.unlink(missing_ok=True)


def operator_result_template(operator: str, operator_key: str) -> dict[str, Any]:
    return {
        "operator": operator,
        "operator_key": operator_key,
        "status": "pending",
        "rounds_attempted": 0,
        "agent_steps_per_round": [],
        "agent_steps_total": 0,
        "tokens": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "rounds": [],
        },
        "compile_only": {"passed": False, "log": ""},
        "verify": {"passed": False, "log": ""},
        "perf": {"passed": False, "log": ""},
        "variants": {"executed": False, "passed": None, "log": ""},
        "final_kernel": "artifact/cuda_/kernel.cu",
        "errors": [],
    }


def run_uniopbench_task(args) -> int:
    config_path = Path(args.config).resolve() if args.config else task_config_path()
    operators_override = args.operators.split(",") if args.operators else None
    task_config = load_task_config(config_path, operators_override=operators_override)

    run_id = args.run_id or timestamp_run_id()
    run_dir = runs_root(task_config.experiment.name) / run_id
    if run_dir.exists() and not args.resume:
        raise FileExistsError(f"Run already exists: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)

    logger = Logger(run_dir / "console.log")
    try:
        summary = load_existing_summary(run_dir) if args.resume else {"operators": {}}
        env = subprocess_env(task_config)

        logger.log(f"Run ID: {run_id}")
        logger.log(f"Config: {config_path}")
        logger.log(f"Benchmark root: {benchmark_root()}")
        if task_config.experiment.enable_torch_compile_baseline:
            logger.log(
                "Note: torch.compile baseline is requested in task.yaml but is not injected into external UniOpBench."
            )

        for operator in task_config.operators:
            operator_key = safe_operator_name(operator)
            if args.resume and operator_key in summary.get("operators", {}):
                status = summary["operators"][operator_key].get("status", "unknown")
                logger.log(f"[skip] {operator}: already recorded (status={status})")
                continue

            source_dir = operator_source_dir(operator)
            op_dir = run_dir / "operators" / operator_key
            artifact_dir = op_dir / "artifact"
            prompt_dir = op_dir / "prompt"
            rounds_dir = op_dir / "rounds"
            prompt_dir.mkdir(parents=True, exist_ok=True)
            rounds_dir.mkdir(parents=True, exist_ok=True)
            copy_operator_tree(source_dir, artifact_dir)

            result = operator_result_template(operator, operator_key)
            result["source_dir"] = str(source_dir)
            result["artifact_dir"] = str(artifact_dir)
            result["supports_variants"] = supports_variants(source_dir / "test.py")

            prior_round: dict[str, Any] | None = None
            # Each operator gets its own workspace (artifact_dir) and isolated agent session/context
            artifact_rel = ""  # workspace is artifact_dir; no cwd override needed
            if args.dry_run:
                system_prompt, user_prompt = build_prompt(
                    task_config, operator, source_dir, None, artifact_rel_path=artifact_rel
                )
                write_text(prompt_dir / "system.txt", system_prompt)
                write_text(prompt_dir / "user.txt", user_prompt)
                result["status"] = "dry_run"
                summary["operators"][operator_key] = result
                write_json(op_dir / "result.json", result)
                write_json(run_dir / "run_summary.json", summary)
                logger.log(f"[dry-run] prepared {operator}")
                continue

            (artifact_dir / ".uniopbench" / "requests").mkdir(parents=True, exist_ok=True)
            for round_idx in range(task_config.experiment.max_repair_rounds + 1):
                round_dir = rounds_dir / f"round_{round_idx}"
                round_dir.mkdir(parents=True, exist_ok=True)
                system_prompt, user_prompt = build_prompt(
                    task_config, operator, source_dir, prior_round, artifact_rel_path=artifact_rel
                )
                write_text(prompt_dir / "system.txt", system_prompt)
                write_text(prompt_dir / "user.txt", user_prompt)

                request_rel_path = Path(".uniopbench") / "requests" / f"round_{round_idx}.md"
                request_path = artifact_dir / request_rel_path
                write_text(request_path, user_prompt)
                live_user_input = (
                    f"Read {request_rel_path.as_posix()} in the workspace root and follow its instructions autonomously. "
                    "Use tools as needed, and reply with a short summary when you finish this turn."
                )

                request_payload = {
                    "provider": task_config.experiment.provider,
                    "model": task_config.experiment.model,
                    "max_tokens": 128_000,
                    "workspace_root": str(artifact_dir),
                    "system_prompt": system_prompt,
                    "user_prompt": user_prompt,
                    "request_file": str(request_path),
                    "live_user_input": live_user_input,
                    "max_agent_steps": _resolve_max_agent_steps(task_config.experiment),
                    "session_mode": "chat",
                    "session_scope": "per_round",
                }
                for key in ("temperature", "top_p", "top_k"):
                    if (val := getattr(task_config.experiment, key)) is not None:
                        request_payload[key] = val

                session_log_path = round_dir / "agent_session.log"
                logger.log(f"[run] {operator} round {round_idx}: launching agent session (tail {session_log_path.relative_to(run_dir)} for live output)...")
                session = LiveAgentSession(
                    task_config=task_config,
                    workspace_dir=artifact_dir,
                    system_prompt_path=prompt_dir / "system.txt",
                    session_log_path=session_log_path,
                )
                try:
                    response_payload = session.run_round(
                        round_dir=round_dir,
                        request_payload=request_payload,
                        live_user_input=live_user_input,
                    )
                finally:
                    session.close()

                write_json(round_dir / "request.json", request_payload)
                write_json(round_dir / "response.json", response_payload)

                usage = response_payload.get("token_usage") or {}
                result["tokens"]["rounds"].append(usage)
                result["tokens"]["prompt_tokens"] += usage.get("prompt_tokens", 0)
                result["tokens"]["completion_tokens"] += usage.get("completion_tokens", 0)
                result["tokens"]["total_tokens"] += usage.get("total_tokens", 0)

                steps_this_round = response_payload.get("steps", 0)
                result["agent_steps_per_round"].append(steps_this_round)
                result["agent_steps_total"] = sum(result["agent_steps_per_round"])

                kernel_source = (artifact_dir / "cuda_" / "kernel.cu").read_text(
                    encoding="utf-8", errors="replace"
                )
                write_text(round_dir / "extracted_kernel.cu", kernel_source)
                result["rounds_attempted"] = round_idx + 1

                compile_code, compile_log = run_command(
                    [sys.executable, "test.py", "--compile-only"],
                    artifact_dir,
                    env,
                    round_dir / "compile.log",
                )
                result["compile_only"] = {
                    "passed": compile_code == 0,
                    "log": str(round_dir / "compile.log"),
                }

                if compile_code == 0:
                    verify_code, verify_log = run_command(
                        [sys.executable, "test.py", "--no-perf"],
                        artifact_dir,
                        env,
                        round_dir / "verify.log",
                    )
                else:
                    verify_code, verify_log = 1, "Skipped because compile-only failed.\n"
                    write_text(round_dir / "verify.log", verify_log)
                result["verify"] = {
                    "passed": verify_code == 0,
                    "log": str(round_dir / "verify.log"),
                }

                if compile_code == 0 and verify_code == 0:
                    perf_code, perf_log = run_command(
                        [sys.executable, "test.py"],
                        artifact_dir,
                        env,
                        round_dir / "perf.log",
                    )
                else:
                    perf_code, perf_log = 1, "Skipped because correctness checks did not pass.\n"
                    write_text(round_dir / "perf.log", perf_log)
                result["perf"] = {
                    "passed": perf_code == 0,
                    "log": str(round_dir / "perf.log"),
                }

                variants_log = ""
                if result["supports_variants"] and compile_code == 0 and verify_code == 0:
                    variants_code, variants_log = run_command(
                        [sys.executable, "test.py", "--variants", "yaml", "--no-perf"],
                        artifact_dir,
                        env,
                        round_dir / "variants.log",
                    )
                    result["variants"] = {
                        "executed": True,
                        "passed": variants_code == 0,
                        "log": str(round_dir / "variants.log"),
                    }
                elif result["supports_variants"]:
                    variants_log = "Skipped because correctness checks did not pass.\n"
                    write_text(round_dir / "variants.log", variants_log)
                    result["variants"] = {
                        "executed": True,
                        "passed": False,
                        "log": str(round_dir / "variants.log"),
                    }
                else:
                    result["variants"] = {
                        "executed": False,
                        "passed": None,
                        "log": "",
                    }

                correctness_ok = result["compile_only"]["passed"] and result["verify"]["passed"]
                if result["variants"]["executed"]:
                    correctness_ok = correctness_ok and bool(result["variants"]["passed"])

                if correctness_ok:
                    result["status"] = "passed"
                    logger.log(f"[pass] {operator} in round {round_idx}")
                    break

                error_message = (
                    f"round {round_idx} failed: compile={compile_code}, "
                    f"verify={verify_code}, perf={perf_code}, variants={result['variants']['passed']}"
                )
                result["errors"].append(error_message)
                logger.log(f"[retry] {operator}: {error_message}")
                prior_round = {
                    "kernel": kernel_source,
                    "compile_log": compile_log,
                    "verify_log": verify_log,
                    "perf_log": perf_log,
                    "variants_log": variants_log,
                }
            else:
                result["status"] = "failed"

            if result["status"] != "passed":
                result["status"] = "failed"
                logger.log(f"[fail] {operator}")
                if task_config.runtime.stop_on_first_failure:
                    if not task_config.runtime.keep_temp_builds:
                        cleanup_artifact_tree(artifact_dir)
                    summary["operators"][operator_key] = result
                    write_json(op_dir / "result.json", result)
                    write_json(run_dir / "run_summary.json", summary)
                    return 1

            if not task_config.runtime.keep_temp_builds:
                cleanup_artifact_tree(artifact_dir)
            summary["operators"][operator_key] = result
            write_json(op_dir / "result.json", result)
            write_json(run_dir / "run_summary.json", summary)

        overall_failed = any(
            item.get("status") == "failed" for item in summary.get("operators", {}).values()
        )
        any_dry_run = any(
            item.get("status") == "dry_run" for item in summary.get("operators", {}).values()
        )
        summary["status"] = "failed" if overall_failed else "dry_run" if any_dry_run else "passed"

        ops = summary.get("operators", {})
        total = len(ops)
        passed = sum(1 for o in ops.values() if o.get("status") == "passed")
        failed_list = [o.get("operator", k) for k, o in ops.items() if o.get("status") == "failed"]
        summary["success_rate"] = f"{passed}/{total}"
        summary["failed"] = failed_list

        write_json(run_dir / "run_summary.json", summary)
        logger.log(f"\nSuccess rate: {summary['success_rate']}")
        if failed_list:
            logger.log(f"Failed: {', '.join(failed_list)}")
        return 1 if overall_failed else 0
    finally:
        logger.close()
