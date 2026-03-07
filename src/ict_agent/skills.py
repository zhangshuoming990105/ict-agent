"""Skill loading and selection."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re


@dataclass
class SkillSpec:
    name: str
    description: str
    tools: list[str]
    triggers: list[str]
    always_on: bool
    instructions: str
    context_mode: str  # "inline" (default) or "fork" (Agent as Skill: run via /run only)


_SESSION_CONTINUATION_HINTS = (
    "session",
    "agent0",
    "agent1",
    "reply",
    "output",
    "read output",
    "read reply",
    "send message",
    "continue",
    "follow up",
    "提问",
    "回复",
    "输出",
    "读取",
    "继续",
    "再问",
    "等待回复",
    "等回复",
    "查看session",
    "其他agent",
)


def _parse_frontmatter(text: str) -> tuple[dict, str]:
    lines = text.splitlines()
    if len(lines) < 3 or lines[0].strip() != "---":
        return {}, text
    meta_lines: list[str] = []
    end = None
    for index in range(1, len(lines)):
        if lines[index].strip() == "---":
            end = index
            break
        meta_lines.append(lines[index])
    if end is None:
        return {}, text

    meta: dict = {}
    key = None
    for raw in meta_lines:
        line = raw.rstrip()
        if not line.strip():
            continue
        if line.lstrip().startswith("- ") and key:
            meta.setdefault(key, []).append(line.lstrip()[2:].strip())
            continue
        if ":" in line:
            key, value = line.split(":", 1)
            key = key.strip()
            value = value.strip()
            meta[key] = [] if value == "" else value
    body = "\n".join(lines[end + 1 :]).strip()
    return meta, body


def load_skills(skills_root: Path) -> dict[str, SkillSpec]:
    skills: dict[str, SkillSpec] = {}
    if not skills_root.exists():
        return skills
    for skill_file in sorted(skills_root.glob("*/SKILL.md")):
        text = skill_file.read_text(encoding="utf-8", errors="replace")
        meta, body = _parse_frontmatter(text)
        name = str(meta.get("name", skill_file.parent.name)).strip()
        if not name:
            continue
        tools = meta.get("tools", [])
        triggers = meta.get("triggers", [])
        if isinstance(tools, str):
            tools = [item.strip() for item in tools.split(",") if item.strip()]
        if isinstance(triggers, str):
            triggers = [item.strip() for item in triggers.split(",") if item.strip()]
        always_on = str(meta.get("always_on", "false")).lower() in ("1", "true", "yes", "on")
        context_raw = str(meta.get("context", "inline")).strip().lower()
        context_mode = "fork" if context_raw == "fork" else "inline"
        skills[name] = SkillSpec(
            name=name,
            description=str(meta.get("description", "")).strip(),
            tools=list(tools),
            triggers=[item.lower() for item in triggers],
            always_on=always_on,
            instructions=body,
            context_mode=context_mode,
        )
    return skills


def select_skills(
    user_input: str,
    all_skills: dict[str, SkillSpec],
    pinned_on: set[str] | None = None,
) -> list[SkillSpec]:
    pinned_on = pinned_on or set()
    text = user_input.lower()
    words = set(re.findall(r"[a-z0-9_./-]+", text))
    path_like = bool(re.search(r"(^|\s)(/|~/)[^\s]*", text)) or bool(
        re.search(r"\b[a-z0-9_.-]+/[a-z0-9_./-]*", text)
    )

    selected: set[str] = set()
    for name, skill in all_skills.items():
        if skill.context_mode == "fork":
            continue
        if skill.always_on or name in pinned_on:
            selected.add(name)
            continue
        if any((trigger in text) or (trigger in words) for trigger in skill.triggers):
            selected.add(name)

    if "filesystem" in all_skills:
        file_intent = any(
            key in text
            for key in (
                "list ",
                "show ",
                "contents",
                "directory",
                "folder",
                "file ",
                "files ",
                "read ",
                "write ",
                "edit ",
                "search ",
                "grep",
                "find ",
            )
        )
        if path_like or file_intent:
            selected.add("filesystem")

    if "shell" in all_skills:
        shell_intent = any(
            key in text
            for key in (
                "run ",
                "execute ",
                "shell",
                "terminal",
                "command",
                "bash",
                "zsh",
                "pwd",
                "ls ",
                "git ",
                "python ",
                "compile",
            )
        ) or bool(re.search(r"[|;&`$()]", text))
        if shell_intent:
            selected.add("shell")

    if "session" in all_skills:
        session_intent = any(hint in text for hint in _SESSION_CONTINUATION_HINTS) or bool(
            re.search(r"\bsession\d+\b", text)
        )
        if session_intent:
            selected.add("session")

    if "core" in all_skills:
        selected.add("core")
    if not selected and all_skills:
        selected.add(next(iter(all_skills.keys())))
    return [all_skills[name] for name in sorted(selected) if name in all_skills and all_skills[name].context_mode != "fork"]


def build_skill_prompt(selected_skills: list[SkillSpec]) -> str:
    if not selected_skills:
        return ""
    lines = ["Active skills guidance:"]
    for skill in selected_skills:
        lines.append(f"- [{skill.name}] {skill.description}")
        if skill.instructions:
            lines.append(skill.instructions.strip())
    return "\n".join(lines).strip()
