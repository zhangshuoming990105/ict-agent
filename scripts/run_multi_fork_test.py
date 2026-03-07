#!/usr/bin/env python3
"""
Multi-fork QA test: run 2, 4, 8, and 16 subagents (qa skill), each answering a subset of 16 questions.
Verifies fork system scales and no threads are left behind.

NOTE: Prefer running the flow manually first (see docs/multi_fork_test_manual.md). Once you have
working prompts and order, we align this script to that flow so automated test doesn't get stuck.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

# Add project root so we can import tests.data
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tests.data.fork_quiz_questions import (
    partition_questions,
    format_task_for_agent,
)


def run(cmd: list[str], check: bool = True, capture: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        cwd=ROOT,
        check=check,
        capture_output=capture,
        text=True,
        timeout=120,
    )


def live_send(session_id: int, message: str) -> None:
    run(
        ["bash", str(ROOT / "scripts/live_session.sh"), "--session-id", str(session_id), "send", message],
        check=True,
    )


def wait_ready(log_path: Path, min_count: int, timeout_sec: int = 300) -> None:
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        try:
            text = log_path.read_text(encoding="utf-8", errors="replace")
        except FileNotFoundError:
            time.sleep(1)
            continue
        count = text.count(">>> Ready for input.")
        if count >= min_count:
            return
        time.sleep(2)
    raise TimeoutError(f"Ready count never reached {min_count} (timeout {timeout_sec}s)")


def count_subagent_qa_in_log(log_path: Path) -> int:
    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return 0
    # Count [subagent qa] or [fork:qa] completion lines
    return text.count("[subagent qa]")


def main() -> int:
    parser = argparse.ArgumentParser(description="Multi-fork QA test (2, 4, 8, 16 subagents)")
    parser.add_argument("--session-id", type=int, default=0, help="Live session id")
    parser.add_argument("--model", default="gpt-oss-120b", help="Model name")
    parser.add_argument("--skip-n", type=int, nargs="*", default=[], help="Skip these N values (e.g. --skip-n 8 16)")
    args = parser.parse_args()

    session_id = args.session_id
    model = args.model
    log_path = ROOT / ".live_session" / f"session_{session_id}" / "stdout.log"
    scripts = ROOT / "scripts"
    failures = []

    # Start session
    run(
        ["bash", str(scripts / "live_session.sh"), "--session-id", str(session_id), "stop"],
        check=False,
    )
    time.sleep(1)
    run(
        [
            "bash", str(scripts / "live_session.sh"),
            "--session-id", str(session_id),
            "start", "--model", model, "--compact-model", model,
        ],
        check=True,
    )
    wait_ready(log_path, 1)

    total_qa_before = 0
    base_ready = 1

    for n in [2, 4, 8, 16]:
        if n in args.skip_n:
            print(f"[run_multi_fork_test] skipping N={n} (--skip-n)")
            continue

        print(f"[run_multi_fork_test] N={n} subagents ...")
        groups = partition_questions(n)
        try:
            base_ready = log_path.read_text(encoding="utf-8", errors="replace").count(">>> Ready for input.")
        except FileNotFoundError:
            pass

        # Send N /fork qa <task> commands
        for i, group in enumerate(groups):
            task = format_task_for_agent(group, i)
            # Keep single line for safe shell passing; truncate if huge
            task_one_line = task.replace("\n", " | ").strip()
            if len(task_one_line) > 2000:
                task_one_line = task_one_line[:1997] + "..."
            live_send(session_id, f"/fork qa {task_one_line}")
            wait_ready(log_path, base_ready + i + 1, timeout_sec=120)

        base_ready = log_path.read_text(encoding="utf-8", errors="replace").count(">>> Ready for input.")

        # One turn to trigger drain and let model see results
        live_send(
            session_id,
            "How many subagent results do you see above? Reply with that number only.",
        )
        wait_ready(log_path, base_ready + 1, timeout_sec=180)

        # Wait for all fork threads and check status
        base_ready = log_path.read_text(encoding="utf-8", errors="replace").count(">>> Ready for input.")
        live_send(session_id, "/fork-wait 90")
        wait_ready(log_path, base_ready + 1, timeout_sec=120)
        live_send(session_id, "/fork-status")
        wait_ready(log_path, base_ready + 2, timeout_sec=30)

        # Assert no leftover threads
        log_text = log_path.read_text(encoding="utf-8", errors="replace")
        if "No fork subagents running" not in log_text and "all finished" not in log_text:
            failures.append(f"N={n}: /fork-status did not report all finished (leftover threads?)")

        # Assert we got n more [subagent qa] results this round (cumulative)
        qa_count = count_subagent_qa_in_log(log_path)
        if qa_count < total_qa_before + n:
            failures.append(
                f"N={n}: expected at least {total_qa_before + n} [subagent qa] in log (got {qa_count})"
            )
        total_qa_before = qa_count

        if not any(f.startswith(f"N={n}:") for f in failures):
            print(f"[run_multi_fork_test] N={n} passed (fork threads finished, {qa_count} qa results total)")

    # Quit and stop
    live_send(session_id, "quit")
    time.sleep(2)
    run(
        ["bash", str(scripts / "live_session.sh"), "--session-id", str(session_id), "stop"],
        check=False,
    )

    if failures:
        for f in failures:
            print(f"[FAIL] {f}", file=sys.stderr)
        print(f"[run_multi_fork_test] failed ({len(failures)}), log: {log_path}", file=sys.stderr)
        return len(failures)
    print(f"[run_multi_fork_test] all passed (N=2,4,8,16). log: {log_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
