"""CUDA domain adapter."""

from __future__ import annotations

from pathlib import Path

from ict_agent.commands.registry import CommandRegistry
from ict_agent.domains.cuda.commands import register_cuda_commands
from ict_agent.domains.cuda.gpu import gpu_status_summary
from ict_agent.domains.cuda.prompts import compose_system_prompt
from ict_agent.domains.cuda.task_manager import (
    get_workspace_path,
    list_tasks,
    load_history_prompt,
    load_task_prompt,
    resolve_task_path,
    save_to_history,
    setup_workspace,
    workspace_summary,
)
from ict_agent.runtime.preemption import set_shell_interrupt_on_preempt


class CudaDomainAdapter:
    def __init__(self, root_dir: Path):
        self.root_dir = root_dir
        self.task_dir: Path | None = None
        self.workspace_root = root_dir
        self.history_prompt = ""
        self.task_prompt = ""
        self.task_context_source = ""
        self.safe_shell_default = False
        self.initial_message = None
        self.system_prompt_override: str | None = None
        self.append_system_prompt: str = ""

    def compose_system_prompt(self) -> str:
        return compose_system_prompt(
            workspace_root=str(self.workspace_root),
            history_prompt=self.history_prompt,
            task_prompt=self.task_prompt,
            use_cuda_domain=self.task_dir is not None,
            system_prompt_override=self.system_prompt_override,
            append_system_prompt=self.append_system_prompt,
        )

    def list_tasks(self) -> str:
        return list_tasks()

    def load_task(self, spec: str, workdir: str | Path | None = None) -> None:
        self.task_dir = resolve_task_path(spec)
        workspace = Path(workdir).resolve() if workdir else self.task_dir / "workdir"
        self.workspace_root = setup_workspace(self.task_dir, workspace)
        self.history_prompt = load_history_prompt(self.task_dir)
        self.task_prompt, task_src = load_task_prompt(self.task_dir)
        self.task_context_source = str(task_src) if task_src else ""
        self.initial_message = "Optimize the PyTorch model in model.py by implementing custom CUDA kernels."

    def reload_task_context(self) -> None:
        if not self.task_dir:
            raise ValueError("No active task")
        self.history_prompt = load_history_prompt(self.task_dir)
        self.task_prompt, task_src = load_task_prompt(self.task_dir)
        self.task_context_source = str(task_src) if task_src else ""

    def inject_task_context(self) -> str:
        if not self.task_dir:
            return ""
        self.task_prompt, task_src = load_task_prompt(self.task_dir)
        self.task_context_source = str(task_src) if task_src else ""
        return self.task_prompt

    def workspace_summary(self) -> str:
        return workspace_summary(self.workspace_root)

    def gpu_status_summary(self) -> str:
        return gpu_status_summary()

    def try_save_history(self, assistant_content: str, ctx, logger) -> None:
        if not self.task_dir:
            return
        workdir = get_workspace_path()
        if not workdir or not (workdir / "model_new.py").is_file():
            return
        try:
            save_to_history(self.task_dir, workdir, profile_result=self.extract_profile_from_text(assistant_content, ctx))
            logger.log(f"  [history] Saved successful implementation to {self.task_dir / 'history'}")
        except Exception as exc:
            logger.log(f"  [history] Failed to save: {exc}")

    @staticmethod
    def extract_profile_from_text(assistant_content: str, ctx) -> dict:
        from ict_agent.domains.cuda.recovery import PROFILE_RESULT_RE

        match = PROFILE_RESULT_RE.search(assistant_content)
        if match:
            try:
                return {
                    "baseline_us": float(match.group(1)),
                    "compile_us": float(match.group(2)),
                    "cuda_us": float(match.group(3)),
                }
            except ValueError:
                return {}
        for msg in reversed(ctx.messages[-10:]):
            content = msg.get("content", "") or ""
            match = PROFILE_RESULT_RE.search(content)
            if match:
                try:
                    return {
                        "baseline_us": float(match.group(1)),
                        "compile_us": float(match.group(2)),
                        "cuda_us": float(match.group(3)),
                    }
                except ValueError:
                    return {}
        return {}

    def handle_preempt_shell_kill(self, command: str, cmd_ctx) -> bool:
        logger = cmd_ctx.logger
        runtime_state = cmd_ctx.runtime_state
        parts = command.strip().lower().split()
        if len(parts) != 3:
            logger.log("\nUsage: /preempt shell-kill on|off\n")
            return True
        arg = parts[2]
        if arg in ("on", "true", "1"):
            runtime_state["preempt_shell_kill"] = True
            set_shell_interrupt_on_preempt(True)
            logger.log("\nPreemption shell-kill: on\n")
            return True
        if arg in ("off", "false", "0"):
            runtime_state["preempt_shell_kill"] = False
            set_shell_interrupt_on_preempt(False)
            logger.log("\nPreemption shell-kill: off\n")
            return True
        logger.log("\nUsage: /preempt shell-kill on|off\n")
        return True

    def register_commands(self, registry: CommandRegistry) -> None:
        register_cuda_commands(registry)
