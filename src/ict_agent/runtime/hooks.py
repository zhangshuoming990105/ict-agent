"""Extension hooks for runtime customization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class RuntimeHooks:
    before_model_call: Any = None
    after_model_call: Any = None
    after_tool_calls: Any = None
