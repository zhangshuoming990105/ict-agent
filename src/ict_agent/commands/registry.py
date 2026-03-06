"""Slash command registry."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


Handler = Callable[[str, "CommandContext"], bool]


@dataclass
class CommandContext:
    client: Any
    ctx: Any
    runtime_state: dict
    logger: Any
    domain_adapter: Any


class CommandRegistry:
    def __init__(self) -> None:
        self._handlers: list[Handler] = []

    def register(self, handler: Handler) -> None:
        self._handlers.append(handler)

    def dispatch(self, command: str, context: CommandContext) -> bool:
        for handler in self._handlers:
            if handler(command, context):
                return True
        return False
