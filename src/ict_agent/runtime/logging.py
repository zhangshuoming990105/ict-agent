"""Run logging utilities."""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path


_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*m")


def strip_ansi(text: str) -> str:
    return _ANSI_ESCAPE_RE.sub("", text)


def is_tty() -> bool:
    """Check if stdout is connected to a terminal (supports colors)."""
    return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


class RunLogger:
    """Tee writes to stdout and an optional plain-text log file.
    
    Automatically applies ANSI color codes when outputting to a TTY,
    and strips them when writing to the log file.
    """

    # ANSI color codes
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    
    # Foreground colors
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"
    GRAY = "\033[90m"
    
    # Semantic level colors
    LEVEL_COLORS = {
        "info": "",                 # no color
        "user": CYAN,              # user input
        "assistant": BOLD + MAGENTA,  # assistant responses (bold + color)
        "tool": BLUE,              # tool calls
        "result": GRAY,             # tool results
        "system": YELLOW,          # system messages
        "error": RED,               # errors
        "success": GREEN,          # success messages
        "debug": DIM,              # debug info
    }

    def __init__(self, log_path: Path | None = None):
        self._file = None
        self._live_file = None  # tee to .live_session/session_<id>/stdout.log when ICT_AGENT_LIVE_LOG set
        self._use_color = is_tty()
        self._streaming_started = False
        if log_path:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            self._file = open(log_path, "w", encoding="utf-8")
        live_log = os.environ.get("ICT_AGENT_LIVE_LOG")
        if live_log:
            try:
                Path(live_log).parent.mkdir(parents=True, exist_ok=True)
                self._live_file = open(live_log, "a", encoding="utf-8", errors="replace")
            except OSError:
                pass

    def is_live_session(self) -> bool:
        """True when stdin is FIFO (agent reads from ict-agent send, not terminal)."""
        return (
            self._live_file is not None
            or os.environ.get("ICT_AGENT_LIVE_SESSION") == "1"
            or bool(os.environ.get("ICT_AGENT_SESSION_TTL"))
        )

    def print_user_prompt(self) -> None:
        """Print colored 'You: ' prefix (no newline) before reading input. Only in TTY; avoids duplicate echo.
        Does not print RESET so terminal echo of user input stays in the same color; call reset_style() after reading.
        In live session mode (stdin=FIFO), print hint to use ict-agent send instead of typing.
        """
        if self.is_live_session():
            print(
                f"{self.LEVEL_COLORS.get('info', '')}Send messages via: ict-agent send <message>{self.RESET}",
                flush=True,
            )
        elif self._use_color:
            print(f"{self.LEVEL_COLORS['user']}You: ", end="", flush=True)

    def reset_style(self) -> None:
        """Reset ANSI style after user input line so following output uses default colors."""
        if self._use_color:
            print(self.RESET, end="", flush=True)

    def log(self, msg: str = "", level: str = "info") -> None:
        """Log a message with optional semantic level for coloring.
        
        Args:
            msg: The message to log
            level: Semantic level (info, user, assistant, tool, result, system, error, success, debug)
        """
        output_msg = msg
        if self._use_color and level in self.LEVEL_COLORS:
            color = self.LEVEL_COLORS[level]
            if color:
                output_msg = f"{color}{msg}{self.RESET}"
        
        # In TTY, skip echoing "You: ..." to stdout (user already saw their input); still write to log file
        if level == "user" and self._use_color:
            if self._file:
                self._file.write(strip_ansi(msg) + "\n")
                self._file.flush()
            if self._live_file:
                self._live_file.write(strip_ansi(msg) + "\n")
                self._live_file.flush()
            return
        print(output_msg)
        if self._file:
            self._file.write(strip_ansi(msg) + "\n")
            self._file.flush()
        if self._live_file:
            self._live_file.write(strip_ansi(msg) + "\n")
            self._live_file.flush()

    def print_streaming(self, text: str) -> None:
        """Print streaming text chunk without newline (for real-time model output).

        On the first chunk of each streaming sequence, emits an
        ``Assistant: `` prefix so log files contain the same marker as
        non-streaming replies.  Call ``end_streaming()`` to reset.
        """
        if not self._streaming_started:
            prefix = "\nAssistant: "
            if self._use_color:
                color = self.LEVEL_COLORS.get("assistant", "")
                sys.stdout.write(f"{color}{prefix}{self.RESET}")
            else:
                sys.stdout.write(prefix)
            if self._file:
                self._file.write(prefix)
            if self._live_file:
                self._live_file.write(prefix)
            self._streaming_started = True
        if self._use_color:
            color = self.LEVEL_COLORS.get("assistant", "")
            sys.stdout.write(f"{color}{text}{self.RESET}")
        else:
            sys.stdout.write(text)
        sys.stdout.flush()
        if self._file:
            self._file.write(strip_ansi(text))
            self._file.flush()
        if self._live_file:
            self._live_file.write(strip_ansi(text))
            self._live_file.flush()

    def end_streaming(self) -> None:
        """Finish a streaming output with a newline and reset the prefix flag."""
        sys.stdout.write("\n")
        sys.stdout.flush()
        if self._file:
            self._file.write("\n")
            self._file.flush()
        if self._live_file:
            self._live_file.write("\n")
            self._live_file.flush()
        self._streaming_started = False

    def close(self) -> None:
        if self._file:
            self._file.close()
            self._file = None
        if self._live_file:
            try:
                self._live_file.close()
            except OSError:
                pass
            self._live_file = None
