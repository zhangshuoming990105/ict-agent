import json
from pathlib import Path
from types import SimpleNamespace

from ict_agent.context import ContextManager
from ict_agent.runtime.agent_loop import process_tool_calls
from ict_agent.tools import set_workspace_root


class DummyLogger:
    def __init__(self):
        self.lines = []

    def log(self, msg="", **kwargs):
        self.lines.append(msg)


def make_tool_response():
    tool_call = SimpleNamespace(
        id="call_1",
        function=SimpleNamespace(name="calculator", arguments='{"expression":"2 + 2"}'),
    )
    message = SimpleNamespace(
        tool_calls=[tool_call],
        model_dump=lambda: {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_1",
                    "function": {"name": "calculator", "arguments": '{"expression":"2 + 2"}'},
                }
            ],
        },
    )
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def test_process_tool_calls_executes_registered_tool():
    ctx = ContextManager("system prompt")
    logger = DummyLogger()
    outcome = process_tool_calls(make_tool_response(), ctx, logger)

    assert outcome.called is True
    assert outcome.failures == []
    assert any("Calling tool: calculator" in line for line in logger.lines)
    assert any(msg.get("role") == "tool" and "2 + 2 = 4" in msg.get("content", "") for msg in ctx.messages)


def make_edit_file_tool_response(path: str, old_text: str, new_text: str):
    args = json.dumps({"path": path, "old_text": old_text, "new_text": new_text})
    tool_call = SimpleNamespace(
        id="call_edit",
        function=SimpleNamespace(name="edit_file", arguments=args),
    )
    message = SimpleNamespace(
        tool_calls=[tool_call],
        model_dump=lambda: {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "call_edit", "function": {"name": "edit_file", "arguments": args}},
            ],
        },
    )
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def test_process_tool_calls_edit_file(tmp_path: Path):
    """Integration: process_tool_calls executes edit_file and modifies file."""
    set_workspace_root(tmp_path)
    f = tmp_path / "code.py"
    f.write_text("def foo():\n    pass\n")

    ctx = ContextManager("system prompt")
    logger = DummyLogger()
    response = make_edit_file_tool_response(
        path="code.py",
        old_text="def foo():\n    pass",
        new_text="def foo():\n    return 42",
    )
    outcome = process_tool_calls(response, ctx, logger)

    assert outcome.called is True
    assert outcome.failures == []
    assert any("Calling tool: edit_file" in line for line in logger.lines)
    assert any(
        msg.get("role") == "tool" and "Successfully" in msg.get("content", "")
        for msg in ctx.messages
    )
    assert f.read_text() == "def foo():\n    return 42\n"
