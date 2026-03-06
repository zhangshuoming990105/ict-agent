from types import SimpleNamespace

from ict_agent.context import ContextManager
from ict_agent.runtime.agent_loop import process_tool_calls


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
