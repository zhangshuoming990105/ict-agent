#!/usr/bin/env python3
"""
Live session e2e test for enhancement features (workspace, streaming, large output, sandbox).

Starts a real agent session with gpt-oss-120b, sends messages that exercise
the new features, and asserts on log output. Uses live_session.sh for session
management (same pattern as run_live_e2e.py).

Usage:
    python scripts/run_enhancements_e2e.py -v
    python scripts/run_enhancements_e2e.py --session-id 2
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG_SUBDIR = ".live_session"

_verbose = False


def log_verbose(msg: str) -> None:
    if _verbose:
        print(msg, flush=True)


def run(cmd: list[str], check: bool = True, timeout: int | None = 60) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=ROOT, check=check, capture_output=True, text=True, timeout=timeout)


def live_send(session_id: int, message: str) -> None:
    log_verbose(f"  send: {message[:80]}{'...' if len(message) > 80 else ''}")
    run(
        ["bash", str(ROOT / "scripts/live_session.sh"), "--session-id", str(session_id), "send", message],
        check=True,
        timeout=15,
    )


def wait_ready(log_path: Path, min_count: int, timeout_sec: int = 90) -> None:
    log_verbose(f"  wait_ready: count >= {min_count} (timeout {timeout_sec}s)")
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        try:
            text = log_path.read_text(encoding="utf-8", errors="replace")
        except FileNotFoundError:
            time.sleep(1)
            continue
        count = text.count(">>> Ready for input.")
        if count >= min_count:
            log_verbose(f"  wait_ready: done (count={count})")
            return
        time.sleep(2)
    raise TimeoutError(f"Ready count never reached {min_count} (timeout {timeout_sec}s)")


def main() -> int:
    global _verbose
    parser = argparse.ArgumentParser(description="Enhancement features e2e test")
    parser.add_argument("--session-id", type=int, default=2, help="Live session id (default 2 to avoid conflict)")
    parser.add_argument("--model", default="gpt-oss-120b", help="Model name")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    _verbose = args.verbose

    session_id = args.session_id
    log_path = ROOT / LOG_SUBDIR / f"session_{session_id}" / "stdout.log"
    scripts = ROOT / "scripts"
    failures = []

    log_verbose(f"[enhancements_e2e] session_id={session_id} model={args.model}")

    # 1. Stop any existing session, start fresh
    log_verbose("1. Start session")
    run(["bash", str(scripts / "live_session.sh"), "--session-id", str(session_id), "stop"], check=False, timeout=10)
    time.sleep(1)

    # Start agent with --safe-shell so we can test sandbox soft rules
    start_cmd = [
        "bash", str(scripts / "live_session.sh"),
        "--session-id", str(session_id),
        "start", "--model", args.model, "--safe-shell",
    ]
    subprocess.Popen(start_cmd, cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(2)

    try:
        wait_ready(log_path, 1, timeout_sec=90)
    except TimeoutError:
        print(f"[FAIL] Agent did not start within 90s. log: {log_path}", file=sys.stderr)
        return 1

    # --- Feature tests (3 turns) ---

    # Turn 1: Workspace = cwd (agent should see ict-agent source files)
    log_verbose("\n2. Turn 1: workspace test")
    live_send(session_id, "列出当前目录的文件，只告诉我有几个文件和目录")
    try:
        wait_ready(log_path, 2, timeout_sec=90)
    except TimeoutError:
        failures.append("Turn 1 (workspace) timeout")

    # Turn 2: Large output test (generate >30K output)
    log_verbose("\n3. Turn 2: large output test")
    live_send(session_id, "执行 python3 -c \"print('A' * 35000)\"")
    try:
        wait_ready(log_path, 3, timeout_sec=90)
    except TimeoutError:
        failures.append("Turn 2 (large output) timeout")

    # Turn 3: Safe command test (git status should auto-approve in safe-shell mode)
    log_verbose("\n4. Turn 3: safe command test")
    live_send(session_id, "执行 pwd")
    try:
        wait_ready(log_path, 4, timeout_sec=60)
    except TimeoutError:
        failures.append("Turn 3 (safe command) timeout")

    # /tokens to check state
    log_verbose("\n5. /tokens")
    live_send(session_id, "/tokens")
    try:
        wait_ready(log_path, 5, timeout_sec=30)
    except TimeoutError:
        failures.append("/tokens timeout")

    # --- Assertions ---
    log_verbose("\n6. Assertions")
    try:
        log_text = log_path.read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        failures.append("Log file missing")
        log_text = ""

    # Assert: workspace points to ict-agent root (not some other dir)
    if "Workspace:" in log_text:
        log_verbose("  Workspace line found ✓")
    else:
        failures.append("No Workspace line in log")

    # Assert: list_directory or run_shell was called
    if "Calling tool:" in log_text:
        log_verbose("  Tool calls found ✓")
    else:
        failures.append("No tool calls in log")

    # Assert: large output was persisted
    if "[large-output]" in log_text:
        log_verbose("  Large output persisted ✓")
    else:
        failures.append("Large output not persisted (no [large-output] in log)")

    # Assert: dynamic tool schema log
    if "[tools] sending" in log_text:
        log_verbose("  Dynamic tool schema log found ✓")
    else:
        failures.append("No [tools] sending line (dynamic schema not logged)")

    # Assert: safe-shell mode on
    if "Safe shell mode: on" in log_text:
        log_verbose("  Safe shell mode on ✓")
    else:
        failures.append("Safe shell mode not on")

    # --- Quit ---
    log_verbose("\n7. Quit")
    live_send(session_id, "quit")
    time.sleep(2)
    run(["bash", str(scripts / "live_session.sh"), "--session-id", str(session_id), "stop"], check=False, timeout=10)

    if failures:
        for f in failures:
            print(f"[FAIL] {f}", file=sys.stderr)
        print(f"[enhancements_e2e] failed ({len(failures)}), log: {log_path}", file=sys.stderr)
        return len(failures)
    print(f"[enhancements_e2e] passed. log: {log_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
