"""Live session smoke test (real API).

Runs a lightweight 3-turn agent session covering: calculator, time,
write_file, run_shell (with sandbox). Uses scripts/run_mixed_e2e.py.

Other live session scripts (run_live_e2e.py, run_fork_smoke.py) are
kept for manual use but not in the CI test suite to save API cost.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

REAL_API_TIMEOUT = int(os.getenv("ICT_AGENT_REAL_API_TIMEOUT", "120"))


def _run_script(script_name: str, timeout: int, session_id: int = 0) -> subprocess.CompletedProcess:
    script = ROOT / "scripts" / script_name
    return subprocess.run(
        [sys.executable, str(script), "--session-id", str(session_id)],
        cwd=ROOT,
        check=False,
        text=True,
        capture_output=True,
        timeout=timeout,
    )


@pytest.mark.real_api
@pytest.mark.skipif(
    os.getenv("ICT_AGENT_RUN_REAL_API") != "1",
    reason="Set ICT_AGENT_RUN_REAL_API=1 to run real API live-session tests.",
)
def test_mixed_e2e_smoke():
    """3-turn live session: calculator+time, write+shell(sandbox), /tokens. Log: .live_session/session_4/stdout.log"""
    result = _run_script("run_mixed_e2e.py", REAL_API_TIMEOUT, session_id=4)
    assert result.returncode == 0, result.stdout + "\n" + result.stderr
