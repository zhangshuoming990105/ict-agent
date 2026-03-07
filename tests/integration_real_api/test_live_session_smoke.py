from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]

# 单次脚本超时（秒）；可用环境变量 ICT_AGENT_REAL_API_TIMEOUT 覆盖
# 若 live 超时或 log 中出现 Error 400：来自 LLM API（如 ksyun），非工具层；log 中凡有 "-> Calling tool:" 的轮次工具均正常返回。
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
def test_live_session_smoke():
    """Run scripts/run_live_e2e.py (5 turns + /tokens). Log: .live_session/session_0/stdout.log"""
    result = _run_script("run_live_e2e.py", REAL_API_TIMEOUT, session_id=0)
    assert result.returncode == 0, result.stdout + "\n" + result.stderr


@pytest.mark.real_api
@pytest.mark.skipif(
    os.getenv("ICT_AGENT_RUN_REAL_API") != "1",
    reason="Set ICT_AGENT_RUN_REAL_API=1 to run real API live-session tests.",
)
def test_fork_skill_smoke():
    """Run /run scout, then main agent uses result; scripts/run_fork_smoke.py. Log: .live_session/session_1/stdout.log"""
    result = _run_script("run_fork_smoke.py", REAL_API_TIMEOUT, session_id=1)
    assert result.returncode == 0, result.stdout + "\n" + result.stderr
