"""Live session control — native Python implementation.

ict-agent start/send/status/stop/paths manage a long-running agent process
whose stdin is a FIFO. External callers (AI agents, scripts) inject messages
via ``ict-agent send`` and read output from ``.live_session/session_<id>/stdout.log``.

The agent process writes to stdout.log via RunLogger's ``_live_file``
(triggered by the ``ICT_AGENT_LIVE_LOG`` environment variable).
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass
class LivePaths:
    state_dir: Path
    fifo_path: Path
    pid_path: Path
    keeper_pid_path: Path
    ttl_pid_path: Path
    log_path: Path


def get_state_paths(root_dir: Path, session_id: str) -> LivePaths:
    state_dir = root_dir / ".live_session" / f"session_{session_id}"
    return LivePaths(
        state_dir=state_dir,
        fifo_path=state_dir / "stdin.fifo",
        pid_path=state_dir / "pid",
        keeper_pid_path=state_dir / "fifo_keeper.pid",
        ttl_pid_path=state_dir / "ttl.pid",
        log_path=state_dir / "stdout.log",
    )


def _read_pid(path: Path) -> int | None:
    if not path.is_file():
        return None
    try:
        raw = path.read_text().strip()
        return int(raw) if raw else None
    except (ValueError, OSError):
        return None


def _is_agent_process(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            cmd = f.read().replace(b"\x00", b" ").decode("utf-8", errors="replace")
    except (OSError, FileNotFoundError):
        return True  # can't read cmdline but process exists
    return "ict-agent" in cmd or "main.py" in cmd or "ict_agent" in cmd


def is_running(paths: LivePaths) -> bool:
    pid = _read_pid(paths.pid_path)
    return pid is not None and _is_agent_process(pid)


# -- helpers ------------------------------------------------------------------

def _start_fifo_keeper(paths: LivePaths) -> None:
    """Hold FIFO write-end open so agent's read doesn't get EOF."""
    code = f"import time; f=open({repr(str(paths.fifo_path))}, 'w'); time.sleep(999999)"
    proc = subprocess.Popen(
        [sys.executable, "-c", code],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    paths.keeper_pid_path.write_text(str(proc.pid))


def _kill_pid_file(path: Path) -> None:
    pid = _read_pid(path)
    if pid is not None:
        try:
            os.kill(pid, signal.SIGKILL)
        except OSError:
            pass
    if path.is_file():
        path.unlink(missing_ok=True)


def _stop_fifo_keeper(paths: LivePaths) -> None:
    _kill_pid_file(paths.keeper_pid_path)


def _stop_ttl_timer(paths: LivePaths) -> None:
    _kill_pid_file(paths.ttl_pid_path)


def _start_ttl_timer(root_dir: Path, session_id: str, ttl_sec: int, paths: LivePaths) -> None:
    """Spawn a background process that stops the session after *ttl_sec* seconds."""
    if ttl_sec <= 0:
        return
    # The timer calls close_session.sh for a graceful stop
    close_script = root_dir / "scripts" / "close_session.sh"
    if close_script.is_file():
        timer_cmd = f"import time; time.sleep({ttl_sec}); import subprocess; subprocess.run(['bash',{repr(str(close_script))},'--session-id',{repr(session_id)}],capture_output=True)"
    else:
        # Fallback: direct kill
        timer_cmd = f"import time,os,signal; time.sleep({ttl_sec}); pid=int(open({repr(str(paths.pid_path))}).read().strip()); os.kill(pid,signal.SIGTERM)"
    proc = subprocess.Popen(
        [sys.executable, "-c", timer_cmd],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    paths.ttl_pid_path.write_text(str(proc.pid))


def _cleanup_stale(paths: LivePaths) -> None:
    for p in (paths.pid_path, paths.keeper_pid_path, paths.ttl_pid_path):
        if p.is_file():
            p.unlink(missing_ok=True)
    if paths.fifo_path.exists():
        paths.fifo_path.unlink(missing_ok=True)


# -- commands -----------------------------------------------------------------

def cmd_start(root_dir: Path, session_id: str, ttl_sec: int, agent_argv: list[str]) -> int:
    """Start a live session: create FIFO, keeper, launch agent with stdin from FIFO."""
    paths = get_state_paths(root_dir, session_id)
    paths.state_dir.mkdir(parents=True, exist_ok=True)

    if is_running(paths):
        print(
            f"A live session is already running for session_id={session_id} "
            f"(pid={_read_pid(paths.pid_path)}).",
            file=sys.stderr,
        )
        return 1

    # Clean stale files
    for p in (paths.fifo_path, paths.log_path, paths.pid_path, paths.keeper_pid_path, paths.ttl_pid_path):
        if p.exists():
            p.unlink(missing_ok=True)

    # Create FIFO and keeper
    os.mkfifo(str(paths.fifo_path))
    _start_fifo_keeper(paths)

    # Build environment: ICT_AGENT_LIVE_LOG triggers RunLogger tee to stdout.log
    env = os.environ.copy()
    env["ICT_AGENT_SESSION_ID"] = session_id
    env["ICT_AGENT_SESSION_TTL"] = str(ttl_sec)
    env["ICT_AGENT_LIVE_LOG"] = str(paths.log_path)
    env["PYTHONUNBUFFERED"] = "1"

    # Open FIFO for reading (agent's stdin)
    fifo_fd = os.open(str(paths.fifo_path), os.O_RDONLY | os.O_NONBLOCK)

    cmd = [sys.executable, "-u", str(root_dir / "main.py")] + agent_argv
    proc = subprocess.Popen(
        cmd,
        stdin=fifo_fd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        cwd=str(root_dir),
        env=env,
        start_new_session=True,
    )
    os.close(fifo_fd)
    paths.pid_path.write_text(str(proc.pid))

    # TTL auto-shutdown timer
    if ttl_sec > 0:
        _start_ttl_timer(root_dir, session_id, ttl_sec, paths)

    print(f"session_id={session_id}")
    print(f"started_pid={proc.pid}")
    print(f"ttl={ttl_sec}")
    print(f"fifo={paths.fifo_path}")
    print(f"log={paths.log_path}")
    return 0


def cmd_send(paths: LivePaths, message: str) -> int:
    if not message:
        print("send requires a non-empty message.", file=sys.stderr)
        return 1
    if not is_running(paths):
        print("No live session running. Start with: ict-agent start --session-id <id>", file=sys.stderr)
        return 1
    with open(paths.fifo_path, "w", encoding="utf-8") as f:
        f.write(message + "\n")
        f.flush()
    print(f"sent: {message}")
    return 0


def cmd_status(paths: LivePaths, session_id: str) -> int:
    if is_running(paths):
        print("running=true")
        print(f"session_id={session_id}")
        print(f"pid={_read_pid(paths.pid_path)}")
    else:
        _stop_fifo_keeper(paths)
        _stop_ttl_timer(paths)
        _cleanup_stale(paths)
        print("running=false")
    print(f"fifo={paths.fifo_path}")
    print(f"log={paths.log_path}")
    return 0


def cmd_stop(paths: LivePaths, session_id: str) -> int:
    if is_running(paths):
        # Graceful: send quit
        try:
            with open(paths.fifo_path, "w", encoding="utf-8") as f:
                f.write("quit\n")
                f.flush()
        except OSError:
            pass

        pid = _read_pid(paths.pid_path)
        deadline = time.time() + 5
        while is_running(paths) and time.time() < deadline:
            time.sleep(0.5)

        # Force kill if still alive
        if is_running(paths) and pid is not None:
            try:
                os.kill(pid, signal.SIGKILL)
            except OSError:
                pass

        _stop_fifo_keeper(paths)
        _stop_ttl_timer(paths)
        _cleanup_stale(paths)
        print("stopped")
    else:
        print("no running session")
        _stop_fifo_keeper(paths)
        _stop_ttl_timer(paths)
        _cleanup_stale(paths)
    return 0


def cmd_paths(paths: LivePaths, session_id: str) -> int:
    print(f"session_id={session_id}")
    print(f"fifo={paths.fifo_path}")
    print(f"log={paths.log_path}")
    print(f"pid_file={paths.pid_path}")
    print(f"keeper_pid_file={paths.keeper_pid_path}")
    print(f"ttl_pid_file={paths.ttl_pid_path}")
    return 0
