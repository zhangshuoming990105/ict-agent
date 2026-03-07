"""Thread-local runtime context for tools that need access to the current agent turn (e.g. fork_subagent)."""

from __future__ import annotations

import threading
from typing import Any

_ctx = threading.local()


def set_current_runtime(
    ctx: Any,
    runtime_state: dict,
    client: Any,
    logger: Any,
) -> None:
    """Set the current turn's context so tools can access it. Call from main loop only."""
    _ctx.ctx = ctx
    _ctx.runtime_state = runtime_state
    _ctx.client = client
    _ctx.logger = logger


def get_current_runtime() -> dict | None:
    """Get the current runtime dict (ctx, runtime_state, client, logger) or None if not set."""
    try:
        return {
            "ctx": getattr(_ctx, "ctx", None),
            "runtime_state": getattr(_ctx, "runtime_state", None),
            "client": getattr(_ctx, "client", None),
            "logger": getattr(_ctx, "logger", None),
        }
    except Exception:
        return None


def clear_current_runtime() -> None:
    """Clear the current runtime. Optional; next set will overwrite."""
    for name in ("ctx", "runtime_state", "client", "logger"):
        if hasattr(_ctx, name):
            delattr(_ctx, name)
