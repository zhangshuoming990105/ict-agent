"""Run logging utilities."""

from __future__ import annotations

import re
from pathlib import Path


_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*m")


def strip_ansi(text: str) -> str:
    return _ANSI_ESCAPE_RE.sub("", text)


class RunLogger:
    """Tee writes to stdout and an optional plain-text log file."""

    def __init__(self, log_path: Path | None = None):
        self._file = None
        if log_path:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            self._file = open(log_path, "w", encoding="utf-8")

    def log(self, msg: str = "") -> None:
        print(msg)
        if self._file:
            self._file.write(strip_ansi(msg) + "\n")
            self._file.flush()

    def close(self) -> None:
        if self._file:
            self._file.close()
            self._file = None
