#!/usr/bin/env python3
"""
Live session e2e test: 5 turns (time, write, read, compile, run) + /tokens.

Session protocol: one message per turn; wait for ">>> Ready for input." before next send.
Use -v/--verbose for step progress.
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

# 5 turns: time tool, write_file, read_file, run_shell (compile), run_shell (run)
TURNS = [
    ("现在几点？用工具查一下", 2),
    ("在 temp/t20 下写一个 hello.c，打印 Hello World", 3),
    ("读 temp/t20/hello.c 确认内容", 4),
    ("用 gcc 编译 temp/t20/hello.c 生成 temp/t20/hello", 5),
    ("运行 temp/t20/hello", 6),
]


def log_verbose(msg: str) -> None:
    if _verbose:
        print(msg, flush=True)


def run(
    cmd: list[str],
    check: bool = True,
    capture: bool = True,
    timeout: int | None = 60,
) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=ROOT, check=check, capture_output=capture, text=True, timeout=timeout)


def live_send(session_id: int, message: str, verbose: bool = False) -> None:
    if verbose:
        preview = message[:80] + "..." if len(message) > 80 else message
        log_verbose(f"  send: {preview}")
    run(
        ["bash", str(ROOT / "scripts/live_session.sh"), "--session-id", str(session_id), "send", message],
        check=True,
        timeout=15,
    )


def wait_ready(log_path: Path, min_count: int, timeout_sec: int = 120, verbose: bool = False) -> None:
    if verbose:
        log_verbose(f"  wait_ready: count >= {min_count} (timeout {timeout_sec}s)")
    deadline = time.monotonic() + timeout_sec
    last_count = -1
    while time.monotonic() < deadline:
        try:
            text = log_path.read_text(encoding="utf-8", errors="replace")
        except FileNotFoundError:
            time.sleep(1)
            continue
        count = text.count(">>> Ready for input.")
        if verbose and count != last_count:
            log_verbose(f"    Ready count = {count}")
            last_count = count
        if count >= min_count:
            if verbose:
                log_verbose(f"  wait_ready: done (count={count})")
            return
        time.sleep(2)
    raise TimeoutError(f"Ready count never reached {min_count} (timeout {timeout_sec}s)")


def main() -> int:
    global _verbose
    parser = argparse.ArgumentParser(description="Live session e2e (5 turns + /tokens)")
    parser.add_argument("--session-id", type=int, default=0, help="Live session id")
    parser.add_argument("--model", default="gpt-oss-120b", help="Model name (use weaker model for robust workflow)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Print each step")
    args = parser.parse_args()
    _verbose = args.verbose

    session_id = args.session_id
    log_path = ROOT / LOG_SUBDIR / f"session_{session_id}" / "stdout.log"
    scripts = ROOT / "scripts"
    failures = []

    log_verbose(f"[run_live_e2e] session_id={session_id} model={args.model} log={log_path}")
    log_verbose("")

    # 1. Start session
    log_verbose("1. Stop any existing, start in background")
    run(["bash", str(scripts / "live_session.sh"), "--session-id", str(session_id), "stop"], check=False, timeout=10)
    time.sleep(1)
    start_cmd = [
        "bash", str(scripts / "live_session.sh"),
        "--session-id", str(session_id),
        "start", "--model", args.model,
    ]
    # compact-model omitted: use CLI default (e.g. gpt-oss-120b)
    subprocess.Popen(start_cmd, cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(2)
    try:
        wait_ready(log_path, 1, timeout_sec=90, verbose=_verbose)
    except TimeoutError:
        print(f"[FAIL] Agent did not reach Ready within 90s. log: {log_path}", file=sys.stderr)
        return 1
    log_verbose("")

    # 2. 5 turns
    log_verbose("2. Run 5 turns")
    for i, (msg, expected) in enumerate(TURNS, start=1):
        if _verbose:
            log_verbose(f"  turn {i}/5 -> ready {expected}")
        live_send(session_id, msg, verbose=_verbose)
        try:
            wait_ready(log_path, expected, timeout_sec=90, verbose=_verbose)
        except TimeoutError:
            failures.append(f"Turn {i} did not reach ready {expected}")
            if not _verbose:
                print(f"[FAIL] Turn {i} timeout", file=sys.stderr)
    log_verbose("")

    # 3. /tokens
    log_verbose("3. /tokens")
    live_send(session_id, "/tokens", verbose=_verbose)
    wait_ready(log_path, 7, timeout_sec=30, verbose=_verbose)
    log_verbose("")

    # 4. Assert
    log_verbose("4. Assert (tools called)")
    try:
        log_text = log_path.read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        failures.append("Log file missing")
        log_text = ""
    ready_count = log_text.count(">>> Ready for input.")
    if ready_count < 7:
        failures.append(f"Ready signals: {ready_count} < 7")
    for tool in ("write_file", "read_file", "run_shell"):
        if f"Calling tool: {tool}" not in log_text:
            failures.append(f"Tool NOT called: {tool}")
    log_verbose("")

    # 5. Quit and stop
    log_verbose("5. Quit and stop session")
    live_send(session_id, "quit", verbose=_verbose)
    time.sleep(2)
    run(["bash", str(scripts / "live_session.sh"), "--session-id", str(session_id), "stop"], check=False, timeout=10)
    log_verbose("   done.")
    log_verbose("")

    if failures:
        for f in failures:
            print(f"[FAIL] {f}", file=sys.stderr)
        print(f"[run_live_e2e] failed ({len(failures)}), log: {log_path}", file=sys.stderr)
        return len(failures)
    print(f"[run_live_e2e] passed. log: {log_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
