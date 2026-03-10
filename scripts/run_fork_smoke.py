#!/usr/bin/env python3
"""
Fork skill smoke: /run scout, then main agent uses result; verify context interaction.

Session protocol: one message per turn; wait for ">>> Ready for input." before next send.
Same pattern as run_multi_fork_test.py. Use -v/--verbose for step progress.
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


def wait_ready(log_path: Path, min_count: int, timeout_sec: int = 90, verbose: bool = False) -> None:
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
    parser = argparse.ArgumentParser(description="Fork skill smoke (/run scout + main context)")
    parser.add_argument("--session-id", type=int, default=0, help="Live session id")
    parser.add_argument("--model", default="gpt-oss-120b", help="Model name")
    parser.add_argument("-v", "--verbose", action="store_true", help="Print each step")
    args = parser.parse_args()
    _verbose = args.verbose

    session_id = args.session_id
    log_path = ROOT / LOG_SUBDIR / f"session_{session_id}" / "stdout.log"
    scripts = ROOT / "scripts"
    failures = []

    log_verbose(f"[run_fork_smoke] session_id={session_id} model={args.model} log={log_path}")
    log_verbose("")

    # 1. Start session
    log_verbose("1. Stop any existing, start in background")
    run(["bash", str(scripts / "live_session.sh"), "--session-id", str(session_id), "stop"], check=False, timeout=10)
    time.sleep(1)
    start_cmd = [
        "bash", str(scripts / "live_session.sh"),
        "--session-id", str(session_id),
        "start", "--model", args.model, "--compact-model", args.model,
    ]
    subprocess.Popen(start_cmd, cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(2)
    try:
        wait_ready(log_path, 1, timeout_sec=90, verbose=_verbose)
    except TimeoutError:
        print(f"[FAIL] Agent did not reach Ready within 90s. log: {log_path}", file=sys.stderr)
        return 1
    log_verbose("")

    # 2. Turn 1: /run scout
    log_verbose("2. Send /run scout ...")
    live_send(
        session_id,
        "/run scout list all Python files under src/ict_agent and report how many",
        verbose=_verbose,
    )
    wait_ready(log_path, 2, timeout_sec=90, verbose=_verbose)
    log_verbose("")

    # 3. Assert fork ran
    try:
        log_text = log_path.read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        log_text = ""
    if "[fork:scout]" not in log_text:
        failures.append("Log does not contain [fork:scout]")
    if "Calling tool: list_directory" not in log_text and "Calling tool: read_file" not in log_text and "Calling tool: search_files" not in log_text:
        failures.append("Log does not show scout tool calls")
    if "Fork result injected" not in log_text and "[subagent scout]" not in log_text:
        failures.append("Log does not show fork result injected or [subagent scout]")
    log_verbose("")

    # 4. Turn 2: main agent uses result
    log_verbose("3. Send follow-up (main agent uses scout result)")
    live_send(
        session_id,
        "Based on the scout result above, how many Python files were reported? Reply briefly with the number or a one-line summary.",
        verbose=_verbose,
    )
    wait_ready(log_path, 3, timeout_sec=90, verbose=_verbose)
    log_verbose("")

    # 5. Assert main reply and Ready
    try:
        log_text = log_path.read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        log_text = ""
    if "Assistant" not in log_text:
        failures.append("Log does not show any Assistant reply")
    if "Ready for input" not in log_text:
        failures.append("Log does not show Ready for input")
    ready_count = log_text.count(">>> Ready for input.")
    if ready_count < 3:
        failures.append(f"Expected >= 3 Ready signals; got {ready_count}")
    log_verbose("")

    # 6. /fork-status (no leftover threads)
    log_verbose("4. Send /fork-status")
    live_send(session_id, "/fork-status", verbose=_verbose)
    wait_ready(log_path, 4, timeout_sec=30, verbose=_verbose)
    try:
        log_text = log_path.read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        log_text = ""
    if "No fork subagents running" not in log_text and "all finished" not in log_text:
        failures.append("Fork status should report no running subagents")
    log_verbose("")

    # 7. Quit and stop
    log_verbose("5. Quit and stop session")
    live_send(session_id, "quit", verbose=_verbose)
    time.sleep(2)
    run(["bash", str(scripts / "live_session.sh"), "--session-id", str(session_id), "stop"], check=False, timeout=10)
    log_verbose("   done.")
    log_verbose("")

    if failures:
        for f in failures:
            print(f"[FAIL] {f}", file=sys.stderr)
        print(f"[run_fork_smoke] failed ({len(failures)}), log: {log_path}", file=sys.stderr)
        return len(failures)
    print(f"[run_fork_smoke] passed. log: {log_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
