"""UniOpBench experiment orchestrator owned by this repository."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*m")


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


def src_root() -> Path:
    return repo_root() / "src"


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
        print(message)
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
    max_repair_rounds: int = 1
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
    has_variants: bool = False,
) -> tuple[str, str]:
    system_parts = [
        "You are operating as the full ict-agent runtime inside an isolated UniOpBench operator workspace.",
        "Use tools to inspect files, edit cuda_/kernel.cu, and run the existing test.py commands autonomously.",
        "Do not modify any file other than cuda_/kernel.cu unless the scaffold itself is broken.",
        "The generated kernel must preserve the existing exported C interface expected by the operator's test.py.",
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
    ]
    if has_variants:
        user_parts.append(f"python test.py --variants yaml --no-perf{cmd_hint}")
    user_parts.append(
        "Work autonomously in this turn. When compile and correctness pass (STATUS: PASSED), "
        "run `python test.py` once for perf, then STOP and reply with a short summary. "
        "Do NOT keep optimizing after correctness passes.",
    )
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


def load_client(provider: str):
    src = str(src_root())
    if src not in sys.path:
        sys.path.insert(0, src)
    from ict_agent.llm import create_client

    return create_client(provider)


def _resolve_max_agent_steps(experiment: ExperimentConfig) -> int:
    """Resolve max_agent_steps: None = formula, 0 = unlimited."""
    if experiment.max_agent_steps is not None:
        return experiment.max_agent_steps
    return max(8, experiment.max_repair_rounds * 4)


def run_agent_round(
    task_config: TaskConfig,
    workspace_dir: Path,
    round_dir: Path,
    system_prompt: str,
    user_prompt: str,
    no_truncate: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    client, provider_name, default_model, base_url = load_client(task_config.experiment.provider)
    model_name = task_config.experiment.model or default_model
    from ict_agent.app.bootstrap import create_command_registry, create_domain_adapter, create_logger
    from ict_agent.runtime.agent_loop import chat
    from ict_agent.tools import set_workspace_root

    # Ensure PYTHONPATH includes UniOpBench root so agent's run_shell can import optest
    _broot = str(benchmark_root())
    _pypath = os.environ.get("PYTHONPATH", "")
    if _broot not in _pypath.split(os.pathsep):
        os.environ["PYTHONPATH"] = _broot + (os.pathsep + _pypath if _pypath else "")
    # Expose task-level env vars so agent's run_shell inherits them
    os.environ["UNIOPBENCH_TASK_CUDA_ARCH"] = task_config.experiment.cuda_arch
    os.environ["UNIOPBENCH_TASK_COMPILE_BASELINE"] = (
        "1" if task_config.experiment.enable_torch_compile_baseline else "0"
    )

    domain_adapter = create_domain_adapter(repo_root())
    domain_adapter.workspace_root = workspace_dir
    set_workspace_root(workspace_dir)
    domain_adapter.task_prompt = (
        "## Runtime Task Context Injection\n"
        "Use this task context for the current autonomous turn.\n\n"
        + system_prompt
    )
    domain_adapter.task_context_source = str(task_template_path())
    logger = create_logger(round_dir / "agent.log")
    request_payload = {
        "provider": provider_name,
        "base_url": base_url,
        "model": model_name,
        "max_tokens": task_config.experiment.max_tokens,
        "workspace_root": str(workspace_dir),
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "max_agent_steps": _resolve_max_agent_steps(task_config.experiment),
    }
    for key in ("temperature", "top_p", "top_k"):
        if (val := getattr(task_config.experiment, key)) is not None:
            request_payload[key] = val
    try:
        result = chat(
            client=client,
            model=model_name,
            max_tokens=128_000,
            max_agent_steps=request_payload["max_agent_steps"],
            safe_shell=False,
            recovery_cleanup=True,
            preempt_shell_kill=False,
            initial_message=user_prompt,
            compact_client=client,
            compact_model=model_name,
            logger=logger,
            command_registry=create_command_registry(domain_adapter),
            domain_adapter=domain_adapter,
            skills_root=repo_root() / "skills",
            no_truncate=no_truncate,
            headless=True,
        )
    finally:
        logger.close()
    response_payload = {
        "assistant_content": result.assistant_content,
        "response_model": result.response_model,
        "steps": result.steps,
        "tool_called": result.tool_called,
        "had_failure": result.had_failure,
        "error": result.error,
        "ctx_messages": result.ctx_messages,
        "token_usage": asdict(result.token_usage),
    }
    if result.ctx_messages:
        from ict_agent.context import ContextManager

        ctx = ContextManager(system_prompt="", max_tokens=128_000)
        ctx.messages = result.ctx_messages
        raw = ctx.format_debug()
        plain = _ANSI_ESCAPE_RE.sub("", raw)
        write_text(round_dir / "trajectory.log", plain)
    return request_payload, response_payload


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


def _extract_tool_blocks(text: str) -> list[str]:
    """Split trajectory text into TOOL (run_shell) blocks."""
    blocks: list[str] = []
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if re.match(r"^\[\d+\] TOOL \(run_shell\)", line):
            block_lines = [line]
            i += 1
            # Collect until next section marker (e.g. [NN] ASSISTANT or [NN] TOOL or --- separator)
            while i < len(lines):
                if re.match(r"^-{10,}$", lines[i]) or re.match(r"^\[\d+\] ", lines[i]):
                    break
                block_lines.append(lines[i])
                i += 1
            blocks.append("\n".join(block_lines))
        else:
            i += 1
    return blocks


def parse_agent_test_results(trajectory_path: Path) -> dict[str, Any] | None:
    """Parse agent trajectory for test.py results run by the agent itself.

    Scans TOOL (run_shell) blocks for test.py executions and extracts pass/fail
    status from the output markers (STATUS: PASSED / STATUS: FAILED).

    Returns a dict with verify_passed, perf_passed, verify_log, perf_log,
    variants_passed, variants_log — or None if no relevant test results found.
    """
    if not trajectory_path.is_file():
        return None
    text = trajectory_path.read_text(encoding="utf-8", errors="replace")
    blocks = _extract_tool_blocks(text)

    verify_passed: bool | None = None
    verify_log = ""
    perf_passed: bool | None = None
    perf_log = ""
    variants_passed: bool | None = None
    variants_log = ""

    for block in blocks:
        # Only consider blocks that actually ran test.py
        cmd_match = re.search(r"command=(.+)", block)
        if not cmd_match:
            continue
        cmd = cmd_match.group(1).strip()
        if "test.py" not in cmd:
            continue

        # Determine status from output markers
        has_passed = "\N{WHITE HEAVY CHECK MARK} STATUS: PASSED" in block or "STATUS: PASSED" in block
        has_failed = "\N{CROSS MARK} STATUS: FAILED" in block or "STATUS: FAILED" in block

        is_compile_only = "--compile-only" in cmd
        is_no_perf = "--no-perf" in cmd and not is_compile_only
        is_variants = "--variants" in cmd
        is_full_run = (
            not is_compile_only
            and not is_no_perf
            and not is_variants
            and re.search(r"python\s+test\.py\s*($|[^-]|2>&1|\|)", cmd)
        )

        # Extract the output portion of the block (after command= line)
        cmd_line_idx = block.find("command=")
        output_text = block[cmd_line_idx:] if cmd_line_idx >= 0 else block

        if is_variants:
            if has_passed and not has_failed:
                variants_passed = True
                variants_log = output_text
            elif has_failed:
                variants_passed = False
                variants_log = output_text
        elif is_no_perf:
            # verify (--no-perf): correctness check
            if has_passed and not has_failed:
                verify_passed = True
                verify_log = output_text
            elif has_failed:
                verify_passed = False
                verify_log = output_text
        elif is_full_run:
            # full run: includes perf
            if has_passed and not has_failed:
                perf_passed = True
                perf_log = output_text
                # Also counts as verify if we haven't seen a dedicated --no-perf pass
                if verify_passed is None:
                    verify_passed = True
                    verify_log = output_text
            elif has_failed:
                perf_passed = False
                perf_log = output_text

    if verify_passed is None and perf_passed is None:
        return None  # No relevant test results found

    return {
        "verify_passed": bool(verify_passed) if verify_passed is not None else False,
        "perf_passed": bool(perf_passed) if perf_passed is not None else False,
        "verify_log": verify_log,
        "perf_log": perf_log,
        "variants_passed": variants_passed,
        "variants_log": variants_log,
    }


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
    no_truncate = getattr(args, "no_truncate", False)

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
                    task_config, operator, source_dir, None, artifact_rel_path=artifact_rel,
                    has_variants=result["supports_variants"],
                )
                write_text(prompt_dir / "system.txt", system_prompt)
                write_text(prompt_dir / "user.txt", user_prompt)
                result["status"] = "dry_run"
                summary["operators"][operator_key] = result
                write_json(op_dir / "result.json", result)
                write_json(run_dir / "run_summary.json", summary)
                logger.log(f"[dry-run] prepared {operator}")
                continue

            for round_idx in range(task_config.experiment.max_repair_rounds + 1):
                round_dir = rounds_dir / f"round_{round_idx}"
                round_dir.mkdir(parents=True, exist_ok=True)
                system_prompt, user_prompt = build_prompt(
                    task_config, operator, source_dir, prior_round, artifact_rel_path=artifact_rel,
                    has_variants=result["supports_variants"],
                )
                write_text(prompt_dir / "system.txt", system_prompt)
                write_text(prompt_dir / "user.txt", user_prompt)

                request_payload, response_payload = run_agent_round(
                    task_config, artifact_dir, round_dir, system_prompt, user_prompt,
                    no_truncate=no_truncate,
                )
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

                # --- Determine correctness: prefer agent's own test results ---
                # The agent may set up env (e.g. LD_LIBRARY_PATH on non-NV GPUs)
                # that the orchestrator's subprocess doesn't inherit, so re-running
                # externally can produce false negatives.  Trust the agent's results
                # when they are available; fall back to external verification otherwise.
                agent_tr = parse_agent_test_results(round_dir / "trajectory.log")
                use_agent_results = agent_tr is not None and agent_tr["verify_passed"]

                if use_agent_results:
                    logger.log(f"  [agent-verified] trusting agent's own test results for {operator}")
                    write_text(round_dir / "verify.log", agent_tr["verify_log"])
                    verify_code = 0
                    verify_log = agent_tr["verify_log"]
                    result["compile_only"] = {
                        "passed": True,
                        "log": str(round_dir / "verify.log"),
                    }
                    result["verify"] = {
                        "passed": True,
                        "log": str(round_dir / "verify.log"),
                    }

                    if agent_tr["perf_log"]:
                        write_text(round_dir / "perf.log", agent_tr["perf_log"])
                        perf_log = agent_tr["perf_log"]
                        result["perf"] = {
                            "passed": agent_tr["perf_passed"],
                            "log": str(round_dir / "perf.log"),
                        }
                    else:
                        perf_log = "Agent did not run performance test.\n"
                        write_text(round_dir / "perf.log", perf_log)
                        result["perf"] = {"passed": False, "log": str(round_dir / "perf.log")}

                    if agent_tr["variants_passed"] is not None:
                        write_text(round_dir / "variants.log", agent_tr["variants_log"])
                        variants_log = agent_tr["variants_log"]
                        result["variants"] = {
                            "executed": True,
                            "passed": agent_tr["variants_passed"],
                            "log": str(round_dir / "variants.log"),
                        }
                    elif result["supports_variants"]:
                        variants_log = ""
                        result["variants"] = {"executed": False, "passed": None, "log": ""}
                    else:
                        variants_log = ""
                        result["variants"] = {"executed": False, "passed": None, "log": ""}
                else:
                    # Fallback: run external verification (original behaviour)
                    verify_code, verify_log = run_command(
                        [sys.executable, "test.py", "--no-perf"],
                        artifact_dir,
                        env,
                        round_dir / "verify.log",
                    )
                    result["compile_only"] = {
                        "passed": verify_code == 0,
                        "log": str(round_dir / "verify.log"),
                    }
                    result["verify"] = {
                        "passed": verify_code == 0,
                        "log": str(round_dir / "verify.log"),
                    }

                    if verify_code == 0:
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
                        "passed": (verify_code == 0 and perf_code == 0),
                        "log": str(round_dir / "perf.log"),
                    }

                    variants_log = ""
                    if result["supports_variants"] and verify_code == 0:
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

                # Overall correctness: verify must pass; variants too if executed.
                correctness_ok = result["verify"]["passed"]
                if result["variants"]["executed"]:
                    correctness_ok = correctness_ok and bool(result["variants"]["passed"])

                if correctness_ok:
                    result["status"] = "passed"
                    logger.log(f"[pass] {operator} in round {round_idx}")
                    break

                error_message = (
                    f"round {round_idx} failed: "
                    f"verify={result['verify']['passed']}, perf={result['perf']['passed']}, "
                    f"variants={result['variants']['passed']}"
                )
                result["errors"].append(error_message)
                logger.log(f"[retry] {operator}: {error_message}")
                prior_round = {
                    "kernel": kernel_source,
                    "compile_log": verify_log,
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