#!/usr/bin/env python3
"""
Multi-fork QA test: follows docs/multi_fork_test_manual.md.

Session protocol: one user message per turn; we must wait for ">>> Ready for input."
before sending the next (no concurrent sends). See docs/async_fork_and_session.md.
Flow: send /fork 1 -> wait Ready -> send /fork 2 -> wait Ready -> wait for log to
show 2 subagent results (drain may inject them while session is idle) -> send
/fork-wait, /fork-status. Asserts: no leftover threads, job_id=1/2 and read_file in log.
Use -v/--verbose to print each step and log poll progress.
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


def wait_ready(log_path: Path, min_count: int, timeout_sec: int = 300, verbose: bool = False) -> None:
    if verbose:
        log_verbose(f"  wait_ready: polling log until '>>> Ready for input.' count >= {min_count} (timeout {timeout_sec}s)")
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


def wait_for_subagent_results(
    log_path: Path, expected_count: int = 2, timeout_sec: int = 240, verbose: bool = False
) -> None:
    """Wait until log contains at least expected_count '[subagent qa job_id=' entries."""
    if verbose:
        log_verbose(f"  wait_for_subagent_results: until '[subagent qa job_id=' count >= {expected_count} (timeout {timeout_sec}s)")
    deadline = time.monotonic() + timeout_sec
    last_count = -1
    while time.monotonic() < deadline:
        try:
            text = log_path.read_text(encoding="utf-8", errors="replace")
        except FileNotFoundError:
            time.sleep(2)
            continue
        count = text.count("[subagent qa job_id=")
        if verbose and count != last_count:
            log_verbose(f"    [subagent qa job_id=] count = {count}")
            last_count = count
        if count >= expected_count:
            if verbose:
                log_verbose(f"  wait_for_subagent_results: done (count={count})")
            return
        time.sleep(3)
    raise TimeoutError(f"Subagent results never reached {expected_count} (timeout {timeout_sec}s)")


def main() -> int:
    global _verbose
    parser = argparse.ArgumentParser(description="Multi-fork QA test (2 subagents, qa + read_file)")
    parser.add_argument("--session-id", type=int, default=0, help="Live session id")
    parser.add_argument("--model", default="gpt-oss-120b", help="Model name")
    parser.add_argument("-v", "--verbose", action="store_true", help="Print each step and log poll progress")
    args = parser.parse_args()
    _verbose = args.verbose

    session_id = args.session_id
    log_path = ROOT / LOG_SUBDIR / f"session_{session_id}" / "stdout.log"
    scripts = ROOT / "scripts"
    failures = []

    log_verbose(f"[run_multi_fork_test] session_id={session_id} model={args.model} log={log_path}")
    log_verbose("")

    # 1. Start session (stop any existing first)
    log_verbose("1. Start session (stop any existing, then start in background)")
    log_verbose("   stop ...")
    run(["bash", str(scripts / "live_session.sh"), "--session-id", str(session_id), "stop"], check=False, timeout=10)
    time.sleep(1)
    start_cmd = [
        "bash", str(scripts / "live_session.sh"),
        "--session-id", str(session_id),
        "start", "--model", args.model, "--compact-model", args.model,
    ]
    log_verbose("   start (Popen, no wait) ...")
    subprocess.Popen(start_cmd, cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(2)
    try:
        wait_ready(log_path, 1, timeout_sec=90, verbose=_verbose)
    except TimeoutError:
        print(f"[FAIL] Agent did not reach Ready within 90s. log: {log_path}", file=sys.stderr)
        return 1
    log_verbose("")

    # 2. Send two /fork qa
    log_verbose("2. Send /fork qa (agent 1: q01..q08)")
    fork1 = "/fork qa Read skills/qa/questions/q01.txt through q08.txt with read_file, answer each, reply with numbered list 1-8."
    live_send(session_id, fork1, verbose=_verbose)
    wait_ready(log_path, 2, timeout_sec=60, verbose=_verbose)
    log_verbose("")
    log_verbose("3. Send /fork qa (agent 2: q09..q16)")
    fork2 = "/fork qa Read skills/qa/questions/q09.txt through q16.txt with read_file, answer each, reply with numbered list 9-16."
    live_send(session_id, fork2, verbose=_verbose)
    wait_ready(log_path, 3, timeout_sec=60, verbose=_verbose)
    log_verbose("")

    # 4. Wait for both subagent results
    log_verbose("4. Wait for 2 subagent results in log (drain may run while session is idle)")
    if not _verbose:
        print("[run_multi_fork_test] waiting for 2 subagent results (up to 300s) ...")
    try:
        wait_for_subagent_results(log_path, expected_count=2, timeout_sec=300, verbose=_verbose)
    except TimeoutError as e:
        failures.append(str(e))
        print(f"[FAIL] {e}", file=sys.stderr)
    log_verbose("")

    # 5. /fork-wait 90, then /fork-status
    log_verbose("5. Send /fork-wait 90")
    live_send(session_id, "/fork-wait 90", verbose=_verbose)
    wait_ready(log_path, 4, timeout_sec=120, verbose=_verbose)
    log_verbose("")
    log_verbose("6. Send /fork-status")
    live_send(session_id, "/fork-status", verbose=_verbose)
    wait_ready(log_path, 5, timeout_sec=30, verbose=_verbose)
    log_verbose("")

    # 7. Assert
    log_verbose("7. Assert (fork-status, job_id=1/2, [fork:qa], read_file)")
    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    if "No fork subagents running" not in log_text and "all finished" not in log_text:
        failures.append("/fork-status did not report all finished (leftover threads?)")
        log_verbose("   FAIL: /fork-status did not report all finished")
    else:
        log_verbose("   ok: fork-status reports all finished")
    n_qa = log_text.count("[subagent qa job_id=")
    if n_qa < 2:
        failures.append("Expected at least 2 [subagent qa job_id=...] in log")
        log_verbose(f"   FAIL: [subagent qa job_id=] count = {n_qa}")
    else:
        log_verbose(f"   ok: [subagent qa job_id=] count = {n_qa}")
    if "[fork:qa]" not in log_text or "read_file" not in log_text:
        failures.append("Log should show [fork:qa] and read_file tool usage")
        log_verbose("   FAIL: missing [fork:qa] or read_file in log")
    else:
        log_verbose("   ok: [fork:qa] and read_file present")
    if "job_id=1" not in log_text or "job_id=2" not in log_text:
        failures.append("Log should contain both job_id=1 and job_id=2")
        log_verbose("   FAIL: missing job_id=1 or job_id=2")
    else:
        log_verbose("   ok: job_id=1 and job_id=2 present")
    log_verbose("")

    # 8. Quit and stop
    log_verbose("8. Quit and stop session")
    live_send(session_id, "quit", verbose=_verbose)
    time.sleep(2)
    run(["bash", str(scripts / "live_session.sh"), "--session-id", str(session_id), "stop"], check=False, timeout=10)
    log_verbose("   done.")
    log_verbose("")

    if failures:
        for f in failures:
            print(f"[FAIL] {f}", file=sys.stderr)
        print(f"[run_multi_fork_test] failed ({len(failures)}), log: {log_path}", file=sys.stderr)
        return len(failures)
    print(f"[run_multi_fork_test] passed. log: {log_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
