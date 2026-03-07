"""Unit tests for fork_subagent and get_subagent_result tools."""

import threading
import time
from pathlib import Path
from queue import Queue
from types import SimpleNamespace

from ict_agent.runtime.current_context import set_current_runtime
from ict_agent.runtime.agent_loop import get_fork_threads_status, wait_for_fork_threads
from ict_agent.skills import load_skills
from ict_agent.tools import execute_tool


def test_fork_subagent_no_runtime():
    result = execute_tool("fork_subagent", '{"skill_name": "scout", "task": "list files"}')
    assert "Error" in result
    assert "runtime" in result.lower()


def test_fork_subagent_unknown_skill():
    skills_root = Path(__file__).resolve().parents[2] / "skills"
    skills = load_skills(skills_root)
    runtime_state = {
        "skills": skills,
        "model": "mco-4",
        "fork_result_queue": Queue(),
        "fork_job_counter": 0,
        "fork_results": {},
    }
    set_current_runtime(ctx=None, runtime_state=runtime_state, client=object(), logger=SimpleNamespace(log=lambda *a, **k: None))
    try:
        result = execute_tool("fork_subagent", '{"skill_name": "nosuchskill", "task": "list files"}')
        assert "Error" in result
        assert "unknown" in result.lower()
    finally:
        from ict_agent.runtime.current_context import clear_current_runtime
        clear_current_runtime()


def test_fork_subagent_non_fork_skill():
    skills_root = Path(__file__).resolve().parents[2] / "skills"
    skills = load_skills(skills_root)
    runtime_state = {
        "skills": skills,
        "model": "mco-4",
        "fork_result_queue": Queue(),
        "fork_job_counter": 0,
        "fork_results": {},
    }
    set_current_runtime(ctx=None, runtime_state=runtime_state, client=object(), logger=SimpleNamespace(log=lambda *a, **k: None))
    try:
        result = execute_tool("fork_subagent", '{"skill_name": "core", "task": "list files"}')
        assert "Error" in result
        assert "fork" in result.lower()
    finally:
        from ict_agent.runtime.current_context import clear_current_runtime
        clear_current_runtime()


def test_get_subagent_result_no_runtime():
    result = execute_tool("get_subagent_result", '{"job_id": "1"}')
    assert "Error" in result
    assert "runtime" in result.lower()


def test_drain_fork_results_injects_into_ctx():
    from ict_agent.context import ContextManager
    from ict_agent.runtime.agent_loop import drain_fork_results

    ctx = ContextManager("system")
    queue = Queue()
    queue.put(("1", "scout", "Found 3 Python files."))
    runtime_state = {"fork_result_queue": queue, "fork_results": {}}
    drain_fork_results(ctx, runtime_state)
    assert len(ctx.messages) == 2
    content = ctx.messages[1].get("content") or ""
    assert "[subagent scout" in content and "job_id=1" in content
    assert runtime_state["fork_results"].get("1") == "Found 3 Python files."


def test_get_fork_threads_status_empty():
    """Fresh runtime or after all threads finished: status is empty."""
    runtime_state = {"fork_threads": []}
    assert get_fork_threads_status(runtime_state) == []


def test_get_fork_threads_status_prunes_finished():
    """Finished threads are pruned; status returns only still-alive."""
    def quick():
        time.sleep(0.05)
    t = threading.Thread(target=quick, daemon=True)
    t.start()
    t.join(timeout=1.0)
    runtime_state = {"fork_threads": [{"job_id": "1", "skill_name": "scout", "thread": t}]}
    assert get_fork_threads_status(runtime_state) == []


def test_wait_for_fork_threads_all_done():
    """wait_for_fork_threads returns True when no threads or all finished."""
    assert wait_for_fork_threads({"fork_threads": []}, timeout_sec=1.0) is True
    t = threading.Thread(target=lambda: None, daemon=True)
    t.start()
    t.join(timeout=1.0)
    assert wait_for_fork_threads({"fork_threads": [{"job_id": "1", "skill_name": "x", "thread": t}]}, timeout_sec=1.0) is True
