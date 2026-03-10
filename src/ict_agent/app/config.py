"""Application configuration models."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AppConfig:
    provider: str
    model: str
    max_tokens: int
    max_agent_steps: int
    safe_shell: bool
    recovery_cleanup: bool
    preempt_shell_kill: bool
    compact_model: str | None = None
