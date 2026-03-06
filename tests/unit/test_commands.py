from types import SimpleNamespace

from ict_agent.commands.common import handle_common_command
from ict_agent.context import ContextManager


class DummyLogger:
    def __init__(self):
        self.lines = []

    def log(self, msg="", **kwargs):
        self.lines.append(msg)


class DummyDomainAdapter:
    def __init__(self):
        self.workspace_root = "/tmp/workspace"

    def workspace_summary(self):
        return "workspace-summary"

    def handle_preempt_shell_kill(self, command, cmd_ctx):
        cmd_ctx.runtime_state["preempt_shell_kill"] = True
        cmd_ctx.logger.log("preempt-shell-kill")
        return True


def test_set_model_command_updates_runtime_state():
    logger = DummyLogger()
    runtime_state = {
        "active_tool_schemas": [],
        "active_skill_prompt": "",
        "verbose": False,
        "model": "mco-4",
        "safe_shell": False,
        "skills": {},
        "pinned_skills": set(),
    }
    command_context = SimpleNamespace(
        client=None,
        ctx=ContextManager("system"),
        runtime_state=runtime_state,
        logger=logger,
        domain_adapter=DummyDomainAdapter(),
    )
    handled = handle_common_command("/set-model gpt-oss-120b", command_context)
    assert handled is True
    assert runtime_state["model"] == "gpt-oss-120b"
    assert any("Model switched" in line for line in logger.lines)
