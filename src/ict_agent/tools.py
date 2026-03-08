"""Tool registry and built-in tools for the refactored runtime."""

from __future__ import annotations

import json
import math
import os
import re
import subprocess
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Callable

from ict_agent.runtime.preemption import is_preempt_requested, shell_interrupt_on_preempt


_TOOL_REGISTRY: dict[str, tuple[Callable, dict]] = {}
_SHELL_SAFE_MODE = True
_SHELL_ALLOWLIST: set[str] = set()
_SHELL_DENYLIST: set[str] = set()
_SHELL_POLICY_LOADED = False
_SANDBOX_ENABLED = False

# Commands known to be safe — auto-approved without prompting (cf. Claude Code permissions.ts)
SAFE_COMMANDS = frozenset({
    "git status", "git diff", "git log", "git branch", "git show",
    "git remote -v", "git stash list",
    "pwd", "ls", "tree", "date", "which", "whoami", "uname -a",
    "cat /etc/os-release", "python --version", "python3 --version",
    "pip list", "pip freeze", "node --version", "npm --version",
})

# Commands/patterns that are always blocked (cf. Claude Code BashTool/prompt.ts BANNED_COMMANDS)
BANNED_COMMAND_PATTERNS = (
    re.compile(r"\brm\s+(-\w+\s+)*-rf\s+/\s*$"),     # rm -rf /
    re.compile(r"\bmkfs\b"),
    re.compile(r"\bdd\s+if=/dev/"),
    re.compile(r":()\s*\{\s*:\|:\s*&\s*\}\s*;"),       # fork bomb
    re.compile(r"\bshutdown\b"),
    re.compile(r"\breboot\b"),
    re.compile(r"\binit\s+0\b"),
)
_CUSTOM_WORKSPACE_ROOT: Path | None = None
_GPU_DEVICE: str | None = None
_GPU_AUTO = False

_GPU_REQUIRED_PATTERNS = (
    r"\butils/compile\.sh\b",
    r"\bpython3?\s+-m\s+utils\.compile\b",
    r"\bpython3?\s+-m\s+utils\.verification\b",
    r"\bpython3?\s+-m\s+utils\.profiling\b",
    r"\bhipcc\b",
    r"\bnvcc\b",
)


def set_gpu_device(device: str | None) -> None:
    global _GPU_DEVICE
    _GPU_DEVICE = device


def set_gpu_auto(enabled: bool) -> None:
    global _GPU_AUTO
    _GPU_AUTO = bool(enabled)


def set_workspace_root(path: str | Path) -> None:
    global _CUSTOM_WORKSPACE_ROOT
    _CUSTOM_WORKSPACE_ROOT = Path(path).resolve()
    reset_shell_policy_cache()


def reset_shell_policy_cache() -> None:
    global _SHELL_POLICY_LOADED
    _SHELL_POLICY_LOADED = False


def _workspace_root() -> Path:
    if _CUSTOM_WORKSPACE_ROOT is not None:
        return _CUSTOM_WORKSPACE_ROOT
    return Path.cwd()


def workspace_root_str() -> str:
    return str(_workspace_root())


def _command_requires_gpu(command: str) -> bool:
    """Return True when a shell command should reserve/configure a GPU.

    Session-management and repo-inspection commands should still be runnable on
    machines with no available GPU. Auto GPU selection is reserved for actual
    compile/verify/profile style commands.
    """
    text = (command or "").strip().lower()
    if not text:
        return False
    return any(re.search(pattern, text) for pattern in _GPU_REQUIRED_PATTERNS)


def _shell_policy_file() -> Path:
    return _workspace_root() / ".shell_policy.json"


def _load_shell_policy_if_needed() -> None:
    global _SHELL_POLICY_LOADED, _SHELL_ALLOWLIST, _SHELL_DENYLIST
    if _SHELL_POLICY_LOADED:
        return
    policy_path = _shell_policy_file()
    _SHELL_ALLOWLIST = set()
    _SHELL_DENYLIST = set()
    if not policy_path.exists():
        _SHELL_POLICY_LOADED = True
        return
    try:
        data = json.loads(policy_path.read_text(encoding="utf-8"))
        allow = data.get("allowlist", [])
        deny = data.get("denylist", [])
        if isinstance(allow, list):
            _SHELL_ALLOWLIST = {str(item) for item in allow if str(item).strip()}
        if isinstance(deny, list):
            _SHELL_DENYLIST = {str(item) for item in deny if str(item).strip()}
    except Exception:
        pass
    finally:
        _SHELL_POLICY_LOADED = True


def _save_shell_policy() -> None:
    policy_path = _shell_policy_file()
    policy_path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "allowlist": sorted(_SHELL_ALLOWLIST),
        "denylist": sorted(_SHELL_DENYLIST),
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }
    policy_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def set_sandbox_enabled(enabled: bool) -> None:
    global _SANDBOX_ENABLED
    _SANDBOX_ENABLED = bool(enabled)


def is_sandbox_enabled() -> bool:
    return _SANDBOX_ENABLED


def set_shell_safety(enabled: bool) -> None:
    global _SHELL_SAFE_MODE
    _SHELL_SAFE_MODE = bool(enabled)
    _load_shell_policy_if_needed()


def _is_banned_command(cmd: str) -> bool:
    """Check if command matches any banned pattern."""
    return any(p.search(cmd) for p in BANNED_COMMAND_PATTERNS)


def _is_safe_command(cmd: str) -> bool:
    """Check if command is in the pre-approved safe list."""
    return cmd in SAFE_COMMANDS


def _matches_allowlist(cmd: str, allowlist: set[str]) -> bool:
    """Check if command matches any allowlist entry (exact or wildcard prefix).

    Supports patterns like "git *" which matches "git status", "git push", etc.
    """
    for pattern in allowlist:
        if pattern.endswith(" *"):
            prefix = pattern[:-2]
            if cmd == prefix or cmd.startswith(prefix + " "):
                return True
        elif cmd == pattern:
            return True
    return False


def _matches_denylist(cmd: str, denylist: set[str]) -> bool:
    """Check if command matches any denylist entry (exact or wildcard prefix)."""
    for pattern in denylist:
        if pattern.endswith(" *"):
            prefix = pattern[:-2]
            if cmd == prefix or cmd.startswith(prefix + " "):
                return True
        elif cmd == pattern:
            return True
    return False


def get_shell_policy_snapshot() -> dict:
    _load_shell_policy_if_needed()
    return {
        "safe_mode": _SHELL_SAFE_MODE,
        "allowlist": sorted(_SHELL_ALLOWLIST),
        "denylist": sorted(_SHELL_DENYLIST),
        "policy_file": str(_shell_policy_file()),
    }


def _resolve_in_workspace(path_str: str) -> Path:
    root = _workspace_root()
    normalized = path_str.strip()
    if normalized in ("/workspace", "/workspace/", "workspace", ".", "./"):
        return root.resolve()
    if normalized.startswith("/workspace/"):
        normalized = normalized[len("/workspace/") :]
    raw = Path(normalized)
    target = raw.resolve() if raw.is_absolute() else (root / raw).resolve()
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"path '{path_str}' is outside workspace root '{root}'") from exc
    return target


def _rel(path: Path) -> str:
    return str(path.relative_to(_workspace_root()))


def tool(name: str, description: str, parameters: dict):
    def decorator(func: Callable) -> Callable:
        schema = {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": parameters,
            },
        }
        _TOOL_REGISTRY[name] = (func, schema)
        return func

    return decorator


def get_all_tool_schemas() -> list[dict]:
    return [schema for _, schema in _TOOL_REGISTRY.values()]


def get_tool_schema_map() -> dict[str, dict]:
    return {name: schema for name, (_, schema) in _TOOL_REGISTRY.items()}


def execute_tool(name: str, arguments: str) -> str:
    if name not in _TOOL_REGISTRY:
        return f"Error: unknown tool '{name}'"
    func, _ = _TOOL_REGISTRY[name]
    try:
        args = json.loads(arguments) if arguments else {}
        result = func(**args)
        return str(result)
    except json.JSONDecodeError as exc:
        return f"Error: invalid JSON arguments: {exc}"
    except TypeError as exc:
        return f"Error: wrong arguments for tool '{name}': {exc}"
    except Exception as exc:
        return f"Error executing tool '{name}': {exc}"


def _confirm_shell_command(command: str) -> str:
    print("\nShell command requested:")
    print(f"  {command}")
    print("Choose an option:")
    print("  1) allow once")
    print("  2) always allow (add to allowlist)")
    print("  3) deny (add to denylist)")
    while True:
        try:
            choice = input("Your choice [1/2/3]: ").strip()
        except EOFError:
            return "deny"
        if choice == "1":
            return "once"
        if choice == "2":
            return "always"
        if choice == "3":
            return "deny"
        print("Please enter 1, 2, or 3.")


def _run_shell_command(command: str, timeout_sec: int, cwd: str | None = None) -> str:
    timeout_sec = max(1, min(timeout_sec, 600))
    work_dir: str | None = None
    if cwd:
        try:
            resolved = _resolve_in_workspace(cwd)
            if resolved.is_dir():
                work_dir = str(resolved)
        except ValueError:
            pass
    if work_dir is None:
        work_dir = str(_workspace_root())

    # Sandbox mode: delegate to sandbox module for process-level isolation
    if _SANDBOX_ENABLED:
        from ict_agent.sandbox import run_sandboxed
        env = os.environ.copy()
        exit_code, stdout, stderr = run_sandboxed(
            command, work_dir, env=env, timeout_sec=timeout_sec,
        )
        lines = [f"exit_code={exit_code}", f"cwd={work_dir}", f"command={command}", "sandbox=enabled"]
        if stdout:
            lines.extend(["stdout:", stdout.strip()])
        if stderr:
            lines.extend(["stderr:", stderr.strip()])
        if not stdout and not stderr:
            lines.append("(no output)")
        return "\n".join(lines)

    env = os.environ.copy()
    needs_gpu = _command_requires_gpu(command)
    if _GPU_DEVICE is not None and needs_gpu:
        env["CUDA_VISIBLE_DEVICES"] = _GPU_DEVICE
        env["HIP_VISIBLE_DEVICES"] = _GPU_DEVICE
    elif _GPU_AUTO and needs_gpu:
        from ict_agent.domains.cuda.gpu import acquire_gpu

        gpu = acquire_gpu()
        if gpu is None:
            return "Error: no available GPU found (all GPUs are busy)"
        env["CUDA_VISIBLE_DEVICES"] = gpu
        env["HIP_VISIBLE_DEVICES"] = gpu

    interrupted = False
    timeout_hit = False
    exit_code = 0
    stdout = ""
    stderr = ""
    started = time.monotonic()

    try:
        with tempfile.TemporaryFile(mode="w+b") as out_f, tempfile.TemporaryFile(mode="w+b") as err_f:
            proc = subprocess.Popen(
                command,
                shell=True,
                stdout=out_f,
                stderr=err_f,
                cwd=work_dir,
                env=env,
                text=False,
            )
            while True:
                code = proc.poll()
                if code is not None:
                    exit_code = int(code)
                    break
                if (time.monotonic() - started) >= timeout_sec:
                    timeout_hit = True
                    proc.terminate()
                    try:
                        proc.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait(timeout=2)
                    exit_code = int(proc.returncode or -9)
                    break
                if shell_interrupt_on_preempt() and is_preempt_requested():
                    interrupted = True
                    proc.terminate()
                    try:
                        proc.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait(timeout=2)
                    exit_code = int(proc.returncode or -15)
                    break
                time.sleep(0.1)
            out_f.flush()
            err_f.flush()
            out_f.seek(0)
            err_f.seek(0)
            stdout = out_f.read().decode("utf-8", errors="replace").strip()
            stderr = err_f.read().decode("utf-8", errors="replace").strip()
    except Exception as exc:
        return f"Error executing shell command: {exc}"

    if timeout_hit:
        return f"Error: command timed out after {timeout_sec}s"

    lines = [
        f"exit_code={exit_code}",
        f"cwd={work_dir}",
        f"command={command}",
    ]
    if interrupted:
        lines.append("interrupted=true")
        lines.append("interrupt_reason=user_preempt")
    if stdout:
        lines.extend(["stdout:", stdout])
    if stderr:
        lines.extend(["stderr:", stderr])
    if not stdout and not stderr:
        lines.append("(no output)")
    return "\n".join(lines)


@tool(
    name="workspace_info",
    description="Return workspace root information for path-aware tool usage.",
    parameters={"type": "object", "properties": {}},
)
def workspace_info() -> str:
    return f"workspace_root={_workspace_root()}"


@tool(
    name="list_directory",
    description="List files and folders in a directory under workspace root.",
    parameters={
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Directory path relative to workspace root."},
            "include_hidden": {"type": "boolean", "description": "Whether to include hidden files."},
            "max_entries": {"type": "integer", "description": "Maximum entries to return (1-500)."},
        },
    },
)
def list_directory(path: str = ".", include_hidden: bool = False, max_entries: int = 200) -> str:
    max_entries = max(1, min(max_entries, 500))
    target = _resolve_in_workspace(path)
    if not target.exists():
        return f"Error: directory not found: {_rel(target)}"
    if not target.is_dir():
        return f"Error: not a directory: {_rel(target)}"
    entries = sorted(target.iterdir(), key=lambda item: (not item.is_dir(), item.name.lower()))
    if not include_hidden:
        entries = [entry for entry in entries if not entry.name.startswith(".")]
    lines = [f"Directory: {_rel(target)}"]
    for entry in entries[:max_entries]:
        kind = "dir " if entry.is_dir() else "file"
        size = "-" if entry.is_dir() else str(entry.stat().st_size)
        lines.append(f"{kind:4}  {size:>8}  {_rel(entry)}")
    if len(entries) > max_entries:
        lines.append(f"... ({len(entries) - max_entries} more entries)")
    return "\n".join(lines)


@tool(
    name="read_file",
    description="Read text file content with line numbers.",
    parameters={
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "File path relative to workspace root."},
            "start_line": {"type": "integer", "description": "1-indexed start line."},
            "max_lines": {"type": "integer", "description": "Maximum number of lines to read (1-500). Use key 'max_lines' only, no leading hyphen."},
        },
        "required": ["path"],
    },
)
def read_file(path: str, start_line: int = 1, max_lines: int = 200) -> str:
    target = _resolve_in_workspace(path)
    if not target.exists():
        return f"Error: file not found: {_rel(target)}"
    if not target.is_file():
        return f"Error: not a file: {_rel(target)}"
    max_lines = max(1, min(max_lines, 500))
    start_line = max(1, start_line)
    try:
        content = target.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:
        return f"Error reading file '{_rel(target)}': {exc}"
    lines = content.splitlines()
    if not lines:
        return f"File is empty: {_rel(target)}"
    start_idx = start_line - 1
    if start_idx >= len(lines):
        return f"Error: start_line {start_line} exceeds file length {len(lines)}"
    end_idx = min(start_idx + max_lines, len(lines))
    out = [f"File: {_rel(target)} (lines {start_line}-{end_idx} of {len(lines)})"]
    for index in range(start_idx, end_idx):
        out.append(f"{index + 1}|{lines[index]}")
    if end_idx < len(lines):
        out.append(f"... ({len(lines) - end_idx} more lines)")
    return "\n".join(out)


@tool(
    name="write_file",
    description="Write text content to a file under workspace root (overwrite or append).",
    parameters={
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "File path relative to workspace root."},
            "content": {"type": "string", "description": "Text content to write."},
            "mode": {"type": "string", "enum": ["overwrite", "append"], "description": "Write mode."},
        },
        "required": ["path", "content"],
    },
)
def write_file(path: str, content: str, mode: str = "overwrite") -> str:
    target = _resolve_in_workspace(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if mode not in ("overwrite", "append"):
        return "Error: mode must be 'overwrite' or 'append'"
    try:
        if mode == "overwrite":
            target.write_text(content, encoding="utf-8")
        else:
            with target.open("a", encoding="utf-8") as handle:
                handle.write(content)
        return (
            f"Wrote {len(content)} chars to {_rel(target)} (mode={mode}) "
            f"at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
    except Exception as exc:
        return f"Error writing file '{_rel(target)}': {exc}"


@tool(
    name="search_files",
    description="Search text in files using ripgrep and return matching lines.",
    parameters={
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "Regex pattern to search for."},
            "path": {"type": "string", "description": "Directory or file path relative to workspace root."},
            "glob": {"type": "string", "description": "Optional file glob."},
            "max_results": {"type": "integer", "description": "Maximum number of matching lines to return (1-200)."},
            "case_sensitive": {"type": "boolean", "description": "Whether the search should be case sensitive."},
        },
        "required": ["pattern"],
    },
)
def search_files(
    pattern: str,
    path: str = ".",
    glob: str | None = None,
    max_results: int = 50,
    case_sensitive: bool = True,
) -> str:
    target = _resolve_in_workspace(path)
    max_results = max(1, min(max_results, 200))
    cmd = ["rg", "-n", pattern, str(target)]
    if glob:
        cmd.extend(["--glob", glob])
    if not case_sensitive:
        cmd.append("-i")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    except FileNotFoundError:
        return _search_files_python(pattern, target, glob, max_results, case_sensitive)
    except Exception as exc:
        return f"Error running search: {exc}"
    if result.returncode not in (0, 1):
        return f"Error searching files: {result.stderr.strip() or result.stdout.strip()}"
    matches = [line for line in result.stdout.splitlines() if line.strip()]
    if not matches:
        return "No matches found."
    out = [f"Search root: {_rel(target) if target.exists() else path}", f"Pattern: {pattern}"]
    out.extend(matches[:max_results])
    if len(matches) > max_results:
        out.append(f"... ({len(matches) - max_results} more matches)")
    return "\n".join(out)


@tool(
    name="grep_text",
    description="Grep-style text search across files. By default pattern is plain text (literal).",
    parameters={
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "Text or regex to search for."},
            "path": {"type": "string", "description": "Directory or file path relative to workspace root."},
            "glob": {"type": "string", "description": "Optional file glob."},
            "max_results": {"type": "integer", "description": "Maximum number of matching lines to return (1-200)."},
            "case_sensitive": {"type": "boolean", "description": "Whether search should be case sensitive."},
            "regex": {"type": "boolean", "description": "If true, pattern is treated as regex."},
        },
        "required": ["pattern"],
    },
)
def grep_text(
    pattern: str,
    path: str = ".",
    glob: str | None = None,
    max_results: int = 50,
    case_sensitive: bool = True,
    regex: bool = False,
) -> str:
    query = pattern if regex else re.escape(pattern)
    return search_files(
        query,
        path=path,
        glob=glob,
        max_results=max_results,
        case_sensitive=case_sensitive,
    )


@tool(
    name="run_shell",
    description="Execute a shell command in the workspace or a subdirectory.",
    parameters={
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "Shell command to execute."},
            "cwd": {"type": "string", "description": "Working directory relative to workspace root."},
            "timeout_sec": {"type": "integer", "description": "Timeout seconds (1-600), default 120."},
        },
        "required": ["command"],
    },
)
def run_shell(command: str, cwd: str | None = None, timeout_sec: int = 120) -> str:
    _load_shell_policy_if_needed()
    cmd = command.strip()
    if not cmd:
        return "Error: command is empty."

    # Always block dangerous commands regardless of mode
    if _is_banned_command(cmd):
        return f"Blocked: command matches a banned pattern and cannot be executed: {cmd}"

    # Always allow safe commands regardless of mode
    if _is_safe_command(cmd):
        return _run_shell_command(cmd, timeout_sec, cwd=cwd)

    if _SHELL_SAFE_MODE:
        if _matches_denylist(cmd, _SHELL_DENYLIST):
            return f"Denied by denylist: {cmd}"
        if _matches_allowlist(cmd, _SHELL_ALLOWLIST):
            return _run_shell_command(cmd, timeout_sec, cwd=cwd)
        decision = _confirm_shell_command(cmd)
        if decision == "deny":
            _SHELL_DENYLIST.add(cmd)
            _SHELL_ALLOWLIST.discard(cmd)
            _save_shell_policy()
            return f"Denied by user (added to denylist): {cmd}"
        if decision == "always":
            _SHELL_ALLOWLIST.add(cmd)
            _SHELL_DENYLIST.discard(cmd)
            _save_shell_policy()
        return _run_shell_command(cmd, timeout_sec, cwd=cwd)
    return _run_shell_command(cmd, timeout_sec, cwd=cwd)


@tool(
    name="shell_policy_status",
    description="Show current shell safety mode and allowlist/denylist status.",
    parameters={"type": "object", "properties": {}},
)
def shell_policy_status() -> str:
    snapshot = get_shell_policy_snapshot()
    lines = [
        f"safe_mode={snapshot['safe_mode']}",
        f"policy_file={snapshot['policy_file']}",
        f"allowlist_count={len(snapshot['allowlist'])}",
        f"denylist_count={len(snapshot['denylist'])}",
    ]
    if snapshot["allowlist"]:
        lines.append("allowlist:")
        lines.extend(f"  - {entry}" for entry in snapshot["allowlist"])
    if snapshot["denylist"]:
        lines.append("denylist:")
        lines.extend(f"  - {entry}" for entry in snapshot["denylist"])
    return "\n".join(lines)


@tool(
    name="calculator",
    description="Evaluate a mathematical expression.",
    parameters={
        "type": "object",
        "properties": {
            "expression": {"type": "string", "description": "The math expression to evaluate."},
        },
        "required": ["expression"],
    },
)
def calculator(expression: str) -> str:
    safe_dict = {
        "sqrt": math.sqrt,
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "log": math.log,
        "log10": math.log10,
        "abs": abs,
        "round": round,
        "pi": math.pi,
        "e": math.e,
    }
    try:
        result = eval(expression, {"__builtins__": {}}, safe_dict)
        return f"{expression} = {result}"
    except Exception as exc:
        return f"Cannot evaluate '{expression}': {exc}"


@tool(
    name="get_current_time",
    description="Get the current date and time.",
    parameters={"type": "object", "properties": {}},
)
def get_current_time() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S (%A)")


@tool(
    name="fork_subagent",
    description=(
        "Start a fork-skill subagent in the background. Use when a task fits a dedicated skill (e.g. scout for read-only file listing). "
        "Result is injected into the conversation when ready, or use get_subagent_result(job_id) to wait in this turn. "
        "return_mode: final = only last reply (default); full_context = entire subagent conversation (for handoff)."
    ),
    parameters={
        "type": "object",
        "properties": {
            "skill_name": {"type": "string", "description": "Name of a fork skill (context: fork), e.g. scout."},
            "task": {"type": "string", "description": "Task description for the subagent."},
            "return_mode": {
                "type": "string",
                "description": "final = return only last reply (saves context); full_context = return full subagent log.",
                "enum": ["final", "full_context"],
            },
        },
        "required": ["skill_name", "task"],
    },
)
def fork_subagent(skill_name: str, task: str, return_mode: str = "final") -> str:
    from ict_agent.runtime.current_context import get_current_runtime
    from ict_agent.runtime.agent_loop import start_async_fork

    if return_mode not in ("final", "full_context"):
        return_mode = "final"
    runtime = get_current_runtime()
    if not runtime or not runtime.get("runtime_state") or not runtime.get("client"):
        return "Error: no runtime context (fork_subagent is only available during an agent turn)."
    runtime_state = runtime["runtime_state"]
    client = runtime["client"]
    logger = runtime["logger"]
    skills = runtime_state.get("skills", {})
    if skill_name not in skills:
        return f"Error: unknown skill '{skill_name}'. Use /skills to list fork skills."
    skill = skills[skill_name]
    if skill.context_mode != "fork":
        return f"Error: '{skill_name}' is not a fork skill (context: fork). Only fork skills can be run as subagents."
    model = runtime_state.get("model", "")
    if not model:
        return "Error: no model set in runtime state."
    job_id = start_async_fork(runtime_state, client, model, skill, task, logger, return_mode=return_mode)
    return f"Started fork job_id={job_id} (skill={skill_name}). Result will be injected when ready, or use get_subagent_result(job_id) to wait."


@tool(
    name="get_subagent_result",
    description="Wait for a fork subagent result by job_id (from fork_subagent). Polls until result is ready or timeout. Drain any completed forks first.",
    parameters={
        "type": "object",
        "properties": {
            "job_id": {"type": "string", "description": "Job ID returned by fork_subagent."},
            "timeout_sec": {"type": "integer", "description": "Max seconds to wait (default 120)."},
        },
        "required": ["job_id"],
    },
)
def get_subagent_result(job_id: str, timeout_sec: int = 120) -> str:
    from ict_agent.runtime.current_context import get_current_runtime
    from ict_agent.runtime.agent_loop import drain_fork_results

    runtime = get_current_runtime()
    if not runtime or not runtime.get("runtime_state") or not runtime.get("ctx"):
        return "Error: no runtime context (get_subagent_result is only available during an agent turn)."
    runtime_state = runtime["runtime_state"]
    ctx = runtime["ctx"]
    deadline = time.monotonic() + max(1, min(timeout_sec, 300))
    while time.monotonic() < deadline:
        # Do not inject into ctx mid-turn: would break tool_use/tool_result ordering (Bedrock etc.)
        drain_fork_results(ctx, runtime_state, inject_into_ctx=False)
        results = runtime_state.get("fork_results", {})
        if job_id in results:
            return results[job_id]
        time.sleep(0.5)
    return f"Timeout waiting for job_id={job_id} after {timeout_sec}s."


def _search_files_python(
    pattern: str,
    target: Path,
    glob: str | None,
    max_results: int,
    case_sensitive: bool,
) -> str:
    flags = 0 if case_sensitive else re.IGNORECASE
    try:
        regex = re.compile(pattern, flags)
    except re.error as exc:
        return f"Error: invalid regex pattern '{pattern}': {exc}"
    if target.is_file():
        candidates = [target] if _is_searchable_text_file(target) else []
    else:
        candidates = [path for path in target.rglob(glob or "*") if path.is_file()]
        candidates = [path for path in candidates if _is_searchable_text_file(path)]
    matches: list[str] = []
    for file_path in candidates:
        try:
            text = file_path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        for index, line in enumerate(text.splitlines(), start=1):
            if regex.search(line):
                matches.append(f"{_rel(file_path)}:{index}:{line}")
                if len(matches) >= max_results:
                    break
        if len(matches) >= max_results:
            break
    if not matches:
        return "No matches found."
    out = [
        "Search backend: python-fallback (rg unavailable)",
        f"Search root: {_rel(target) if target.exists() else str(target)}",
        f"Pattern: {pattern}",
    ]
    out.extend(matches)
    return "\n".join(out)


def _is_searchable_text_file(path: Path) -> bool:
    if "__pycache__" in path.parts:
        return False
    if path.suffix.lower() in {".pyc", ".pyo", ".so", ".dylib", ".exe", ".bin", ".o"}:
        return False
    try:
        with path.open("rb") as handle:
            sample = handle.read(2048)
        if b"\x00" in sample:
            return False
    except Exception:
        return False
    return True
