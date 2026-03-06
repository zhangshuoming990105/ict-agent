"""Application bootstrap helpers."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from ict_agent.app.config import AppConfig
from ict_agent.commands.common import register_common_commands
from ict_agent.commands.registry import CommandRegistry
from ict_agent.domains.cuda.adapter import CudaDomainAdapter
from ict_agent.runtime.logging import RunLogger


def create_domain_adapter(root_dir: Path) -> CudaDomainAdapter:
    return CudaDomainAdapter(root_dir=root_dir)


def create_command_registry(domain_adapter: CudaDomainAdapter) -> CommandRegistry:
    registry = CommandRegistry()
    register_common_commands(registry)
    domain_adapter.register_commands(registry)
    return registry


def summarize_config(config: AppConfig) -> dict:
    return asdict(config)


def create_logger(log_path):
    return RunLogger(log_path)
