#!/usr/bin/env python3
"""
Live session e2e: mixed feature combo test (lightweight).

3 turns covering core tools, filesystem, shell+sandbox, and /tokens.

Usage:
    python scripts/run_mixed_e2e.py -v
    python scripts/run_mixed_e2e.py --session-id 4 --model gpt-oss-120b -v
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


def vlog(msg: str) -> None:
    if _verbose:
        print(msg, flush=True)


def run(cmd: list[str], check: bool = True, timeout: int = 60) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=ROOT, check=check, capture_output=True, text=True, timeout=timeout)


def live_send(session_id: int, message: str) -> None:
    preview = message[:80] + "..." if len(message) > 80 else message
    vlog(f"  >>> send: {preview}")
    run(
        ["bash", str(ROOT / "scripts/live_session.sh"), "--session-id", str(session_id), "send", message],
        check=True, timeout=15,
    )


def wait_ready(log_path: Path, min_count: int, timeout_sec: int = 90) -> None:
    vlog(f"  ... wait_ready >= {min_count} (timeout {timeout_sec}s)")
    deadline = time.monotonic() + timeout_sec
    last = -1
    while time.monotonic() < deadline:
        try:
            text = log_path.read_text(encoding="utf-8", errors="replace")
        except FileNotFoundError:
            time.sleep(1)
            continue
        count = text.count(">>> Ready for input.")
        if count != last:
            vlog(f"      ready count = {count}")
            last = count
        if count >= min_count:
            return
        time.sleep(2)
    raise TimeoutError(f"Ready count never reached {min_count} (timeout {timeout_sec}s)")


# 3 turns: calculator+time, write+run (sandbox), /tokens
TURNS = [
    (
        "calculator + time",
        "用计算器算 123 * 456，然后告诉我现在几点",
    ),
    (
        "write + shell run (sandbox)",
        "写一个 temp/mixed_test/hi.py 内容是 print('hello')，然后用 run_shell 执行它",
    ),
    (
        "/tokens",
        "/tokens",
    ),
]


def main() -> int:
    global _verbose
    parser = argparse.ArgumentParser(description="Mixed feature combo e2e test")
    parser.add_argument("--session-id", type=int, default=4, help="Session id (default 4)")
    parser.add_argument("--model", default="gpt-oss-120b", help="Model name")
    parser.add_argument("--sandbox", action="store_true", default=True, help="Enable sandbox")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    _verbose = args.verbose

    sid = args.session_id
    log_path = ROOT / LOG_SUBDIR / f"session_{sid}" / "stdout.log"
    scripts = ROOT / "scripts"
    failures = []

    vlog(f"[mixed_e2e] session_id={sid} model={args.model} sandbox={args.sandbox}")
    vlog(f"[mixed_e2e] log: {log_path}")
    vlog("")

    # 1. Start session
    vlog("== Phase 1: Start session ==")
    run(["bash", str(scripts / "live_session.sh"), "--session-id", str(sid), "stop"], check=False, timeout=10)
    time.sleep(1)

    start_cmd = [
        "bash", str(scripts / "live_session.sh"),
        "--session-id", str(sid),
        "start", "--model", args.model,
    ]
    if args.sandbox:
        start_cmd.append("--sandbox")

    subprocess.Popen(start_cmd, cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(2)

    try:
        wait_ready(log_path, 1, timeout_sec=90)
    except TimeoutError:
        print(f"[FAIL] Agent did not start within 90s. log: {log_path}", file=sys.stderr)
        return 1
    vlog("")

    # 2. Send turns
    vlog("== Phase 2: Send turns ==")
    ready_count = 1
    for i, (label, msg) in enumerate(TURNS, start=1):
        ready_count += 1
        vlog(f"\n-- Turn {i}/{len(TURNS)}: {label} --")
        live_send(sid, msg)
        try:
            wait_ready(log_path, ready_count, timeout_sec=120)
        except TimeoutError:
            failures.append(f"Turn {i} ({label}) timeout")
            vlog(f"  [TIMEOUT] turn {i}")
    vlog("")

    # 3. Assertions
    vlog("== Phase 3: Assertions ==")
    try:
        log_text = log_path.read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        failures.append("Log file missing")
        log_text = ""

    checks = [
        ("calculator called", "Calling tool: calculator"),
        ("get_current_time called", "Calling tool: get_current_time"),
        ("write_file called", "Calling tool: write_file"),
        ("run_shell called", "Calling tool: run_shell"),
        ("sandbox enabled", "sandbox=enabled"),
        ("workspace line", "Workspace:"),
        ("dynamic tool schema", "[tools] sending"),
    ]

    for label, check in checks:
        if check in log_text:
            vlog(f"  [OK] {label}")
        else:
            failures.append(f"Assertion failed: {label}")
            vlog(f"  [FAIL] {label}")

    final_ready = log_text.count(">>> Ready for input.")
    expected_ready = len(TURNS) + 1
    vlog(f"\n  Ready signals: {final_ready} (expected >= {expected_ready})")
    if final_ready < expected_ready:
        failures.append(f"Ready count {final_ready} < expected {expected_ready}")

    vlog("")

    # 4. Quit
    vlog("== Phase 4: Quit ==")
    live_send(sid, "quit")
    time.sleep(3)
    run(["bash", str(scripts / "live_session.sh"), "--session-id", str(sid), "stop"], check=False, timeout=10)
    vlog("   session stopped.")
    vlog("")

    if failures:
        for f in failures:
            print(f"[FAIL] {f}", file=sys.stderr)
        print(f"\n[mixed_e2e] FAILED ({len(failures)} failures). log: {log_path}", file=sys.stderr)
        return len(failures)
    print(f"[mixed_e2e] ALL PASSED. log: {log_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
