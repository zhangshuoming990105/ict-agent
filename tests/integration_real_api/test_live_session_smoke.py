from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.real_api
@pytest.mark.skipif(
    os.getenv("ICT_AGENT_RUN_REAL_API") != "1",
    reason="Set ICT_AGENT_RUN_REAL_API=1 to run real API live-session tests.",
)
def test_live_session_smoke():
    script = ROOT / "scripts" / "run_live_e2e.sh"
    result = subprocess.run(
        ["bash", str(script)],
        cwd=ROOT,
        check=False,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stdout + "\n" + result.stderr
