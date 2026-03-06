"""CUDA task workspace lifecycle management."""

from __future__ import annotations

from datetime import datetime
import json
import re
import shutil
from pathlib import Path


PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parents[3]
LOCAL_TEMPLATE_DIR = PROJECT_ROOT / "template"
LOCAL_TASK_DIR = PROJECT_ROOT / "task"
LOCAL_DATASET_DIR = PROJECT_ROOT / "dataset"
LEGACY_ROOT = PROJECT_ROOT.parent / "08_preemptible_cuda_agent"
LEGACY_TEMPLATE_DIR = LEGACY_ROOT / "template"
LEGACY_TASK_DIR = LEGACY_ROOT / "task"
LEGACY_DATASET_DIR = LEGACY_ROOT / "dataset"
TASK_CONTEXT_FILES = ("task.md", "TASK.md")
GLOBAL_TASK_TEMPLATE_FILES = (
    "task/TASK_TEMPLATE.md",
    "task/task.md",
    "TASK_TEMPLATE.md",
    "task.md",
)

_active_workspace: Path | None = None
_dataset_index: dict | None = None
_level_counts: list[tuple[str, int]] | None = None


def _template_dir() -> Path:
    return LOCAL_TEMPLATE_DIR if LOCAL_TEMPLATE_DIR.is_dir() else LEGACY_TEMPLATE_DIR


def _task_dir() -> Path:
    return LOCAL_TASK_DIR if LOCAL_TASK_DIR.is_dir() else LEGACY_TASK_DIR


def _dataset_dir() -> Path:
    return LOCAL_DATASET_DIR if LOCAL_DATASET_DIR.is_dir() else LEGACY_DATASET_DIR


def _load_index() -> dict:
    global _dataset_index, _level_counts
    if _dataset_index is not None:
        return _dataset_index
    index_file = _dataset_dir() / "index.json"
    if not index_file.is_file():
        _dataset_index = {}
        _level_counts = []
        return _dataset_index
    with index_file.open(encoding="utf-8") as handle:
        _dataset_index = json.load(handle)
    counts: dict[str, int] = {}
    for key in _dataset_index:
        level = key.split("/")[0]
        counts[level] = counts.get(level, 0) + 1
    _level_counts = [(level, counts[level]) for level in sorted(counts)]
    return _dataset_index


def _get_level_counts() -> list[tuple[str, int]]:
    _load_index()
    return _level_counts or []


def resolve_task_path(spec: str) -> Path:
    spec = spec.strip()
    candidate = Path(spec)
    if not candidate.is_absolute():
        candidate = (_task_dir() / spec).resolve()
    if candidate.is_dir() and (candidate / "model.py").is_file():
        return candidate

    match = re.match(r"^(level\d+)/(\d+)$", spec)
    if match:
        level, num = match.group(1), int(match.group(2))
        task_dir = _dataset_dir() / level / f"{num:03d}"
        if task_dir.is_dir() and (task_dir / "model.py").is_file():
            return task_dir
        raise FileNotFoundError(f"Dataset task not found: {level}/{num:03d}")

    if spec.isdigit():
        global_id = int(spec)
        if global_id < 1:
            raise ValueError(f"Global task ID must be >= 1, got {global_id}")
        offset = global_id
        for level, count in _get_level_counts():
            if offset <= count:
                task_dir = _dataset_dir() / level / f"{offset:03d}"
                if task_dir.is_dir() and (task_dir / "model.py").is_file():
                    return task_dir
                raise FileNotFoundError(f"Dataset task not found: {level}/{offset:03d}")
            offset -= count
        total = sum(count for _, count in _get_level_counts())
        raise ValueError(f"Global task ID {global_id} out of range (max {total})")

    raise FileNotFoundError(
        f"Cannot resolve task spec '{spec}': not a directory, not a level/id, not a number"
    )


def list_tasks(level_filter: str | None = None) -> str:
    index = _load_index()
    if not index:
        return "No dataset found. Populate ict-agent/dataset or keep using the Step 08 dataset."
    lines: list[str] = []
    current_level = ""
    count = 0
    for task_id in sorted(index.keys()):
        level = task_id.split("/")[0]
        if level_filter and level != level_filter:
            continue
        if level != current_level:
            if current_level:
                lines.append("")
            current_level = level
            level_total = sum(1 for key in index if key.startswith(level + "/"))
            lines.append(f"=== {level} ({level_total} tasks) ===")
        entry = index[task_id]
        desc = entry.get("description", "")[:60]
        score = entry.get("score", "?")
        lines.append(f"  {task_id}  score={score}  {desc}")
        count += 1
    lines.append("")
    global_info = []
    running = 0
    for level, level_count in _get_level_counts():
        global_info.append(f"{level}: global {running + 1}-{running + level_count}")
        running += level_count
    lines.append(f"Total: {count} tasks  ({', '.join(global_info)})")
    return "\n".join(lines)


def setup_workspace(task_dir: str | Path, workdir: str | Path) -> Path:
    global _active_workspace
    task_dir = Path(task_dir).resolve()
    workdir = Path(workdir).resolve()
    model_file = task_dir / "model.py"
    if not model_file.is_file():
        raise FileNotFoundError(f"Task directory missing model.py: {task_dir}")
    if workdir.exists():
        shutil.rmtree(workdir)
    workdir.mkdir(parents=True)
    for item in _template_dir().iterdir():
        dest = workdir / item.name
        if item.is_dir():
            shutil.copytree(item, dest)
        else:
            shutil.copy2(item, dest)
    shutil.copy2(model_file, workdir / "model.py")
    (workdir / "kernels").mkdir(exist_ok=True)
    _active_workspace = workdir
    return workdir


def get_workspace_path() -> Path | None:
    return _active_workspace


def workspace_summary(workdir: str | Path | None = None) -> str:
    root = Path(workdir) if workdir else _active_workspace
    if root is None or not root.exists():
        return "(no active workspace)"
    lines = [f"Workspace: {root}"]
    for item in sorted(root.iterdir()):
        if item.name.startswith(".") or item.name == "__pycache__":
            continue
        if item.is_dir():
            children = [child.name for child in sorted(item.iterdir()) if not child.name.startswith(".")]
            lines.append(f"  {item.name}/  ({len(children)} items: {', '.join(children[:8])})")
        else:
            lines.append(f"  {item.name}  ({item.stat().st_size} bytes)")
    return "\n".join(lines)


def save_to_history(task_dir: str | Path, workdir: str | Path, profile_result: dict | None = None) -> Path:
    task_dir = Path(task_dir).resolve()
    workdir = Path(workdir).resolve()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    entry = task_dir / "history" / timestamp
    entry.mkdir(parents=True, exist_ok=True)
    model_new = workdir / "model_new.py"
    if model_new.is_file():
        shutil.copy2(model_new, entry / "model_new.py")
    kernels_src = workdir / "kernels"
    if kernels_src.is_dir():
        kernels_dst = entry / "kernels"
        if kernels_dst.exists():
            shutil.rmtree(kernels_dst)
        shutil.copytree(
            kernels_src,
            kernels_dst,
            ignore=shutil.ignore_patterns("*.hip", "*_hip.cpp", "*.o"),
        )
    result = {
        "verify": "pass",
        "timestamp": timestamp,
        "profile": profile_result or {},
    }
    (entry / "result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return entry


def load_history_prompt(task_dir: str | Path) -> str:
    task_dir = Path(task_dir).resolve()
    history_dir = task_dir / "history"
    if not history_dir.is_dir():
        return ""
    entries = sorted([path for path in history_dir.iterdir() if path.is_dir()], key=lambda path: path.name, reverse=True)
    if not entries:
        return ""
    latest = entries[0]
    lines = [
        "## Previous Successful Implementation (for reference)",
        "A prior run on this task produced a working implementation. "
        "Use it as a starting point or reference.",
    ]
    result_file = latest / "result.json"
    if result_file.is_file():
        try:
            result = json.loads(result_file.read_text(encoding="utf-8"))
            profile = result.get("profile", {})
            if profile:
                lines.append(
                    f"\nPrior profiling: Baseline={profile.get('baseline_us', '?')}us, "
                    f"Compile={profile.get('compile_us', '?')}us, "
                    f"CUDA={profile.get('cuda_us', '?')}us"
                )
        except (json.JSONDecodeError, OSError):
            pass
    model_new_file = latest / "model_new.py"
    if model_new_file.is_file():
        code = model_new_file.read_text(encoding="utf-8", errors="replace")
        lines.append(f"\nPrior model_new.py:\n```python\n{code.strip()}\n```")
    kernels_dir = latest / "kernels"
    if kernels_dir.is_dir():
        for kernel_file in sorted(kernels_dir.iterdir())[:3]:
            if kernel_file.suffix in (".cu", ".cpp") and kernel_file.is_file():
                code = kernel_file.read_text(encoding="utf-8", errors="replace")
                if len(code) > 4000:
                    code = code[:4000] + "\n... (truncated)"
                lines.append(f"\nPrior {kernel_file.name}:\n```cpp\n{code.strip()}\n```")
    return "\n".join(lines)


def find_task_context_file(task_dir: str | Path) -> Path | None:
    task_dir = Path(task_dir).resolve()
    for name in TASK_CONTEXT_FILES:
        candidate = task_dir / name
        if candidate.is_file():
            return candidate
    return None


def load_task_prompt(task_dir: str | Path) -> tuple[str, Path | None]:
    src = find_task_context_file(task_dir)
    if src is None:
        for rel in GLOBAL_TASK_TEMPLATE_FILES:
            for base in (_task_dir(), PROJECT_ROOT):
                candidate = (base / rel).resolve()
                if candidate.is_file():
                    src = candidate
                    break
            if src is not None:
                break
        if src is None:
            return "", None
    try:
        text = src.read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return "", src
    if not text:
        return "", src
    lines = [
        "## Task Context (from task markdown)",
        "The following task guidance is user-authored ground truth for this task.",
        "",
        f"Source: {src}",
        "",
        text,
    ]
    return "\n".join(lines), src
