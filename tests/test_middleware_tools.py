import asyncio

import pytest
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

from evenkeel.failure import BadInputError
from evenkeel.middleware import EvenKeelMiddleware
from tests.fakes import ScriptedChatModel, ai, tool_call


class RateLimited(Exception):
    status_code = 429


def build(tools, script, thread, **kwargs):
    model = ScriptedChatModel(script=script)
    agent = create_agent(
        model, tools=tools, middleware=[EvenKeelMiddleware(**kwargs)], checkpointer=InMemorySaver()
    )
    return model, agent, {"configurable": {"thread_id": thread}}


def test_transient_tool_errors_retry_the_tool():
    calls = {"n": 0}

    @tool
    def send_email(to: str) -> str:
        """Send an email."""
        calls["n"] += 1
        if calls["n"] == 1:
            raise RateLimited("429 Too Many Requests")
        return "sent"

    script = [ai(tool_calls=[tool_call("send_email", {"to": "a@b.c"})]), ai("done")]
    model, agent, config = build([send_email], script, "tool-retry", sleep=lambda s: None)
    result = agent.invoke({"messages": [HumanMessage("hi")]}, config)
    tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert calls["n"] == 2 and tool_messages[0].content == "sent"
    attempts = agent.get_state(config).values["evenkeel_attempts"]
    assert [(a["call"], a["kind"], a["action"]) for a in attempts] == [
        ("tool:send_email", "transient", "retry")
    ]


def test_bad_input_goes_back_to_the_model_as_an_error_tool_message():
    @tool
    def lookup(order_id: str) -> str:
        """Look up an order."""
        raise BadInputError(f"order id {order_id!r} has the wrong format")

    script = [ai(tool_calls=[tool_call("lookup", {"order_id": "x"})]), ai("done")]
    model, agent, config = build([lookup], script, "tool-bad-input")
    result = agent.invoke({"messages": [HumanMessage("hi")]}, config)
    error = next(m for m in result["messages"] if isinstance(m, ToolMessage))
    assert error.status == "error" and "wrong format" in error.content
    sent_back = model.calls[1][-1]
    assert isinstance(sent_back, ToolMessage) and sent_back.status == "error" and len(model.calls) == 2
    assert agent.get_state(config).values["evenkeel_attempts"][0]["action"] == "returned_to_model"


def test_fatal_tool_errors_propagate():
    @tool
    def broken(x: str) -> str:
        """Broken."""
        raise TypeError("bug in the tool")

    script = [ai(tool_calls=[tool_call("broken", {"x": "1"})])]
    model, agent, config = build([broken], script, "tool-fatal")
    with pytest.raises(TypeError, match="bug in the tool"):
        agent.invoke({"messages": [HumanMessage("hi")]}, config)


def test_async_tool_retry():
    calls = {"n": 0}

    @tool
    async def fetch(url: str) -> str:
        """Fetch a page."""
        calls["n"] += 1
        if calls["n"] == 1:
            raise TimeoutError("timed out")
        return "page"

    async def asleep(seconds: float) -> None:
        return None

    script = [ai(tool_calls=[tool_call("fetch", {"url": "u"})]), ai("done")]
    model, agent, config = build([fetch], script, "tool-async", asleep=asleep)
    result = asyncio.run(agent.ainvoke({"messages": [HumanMessage("hi")]}, config))
    assert calls["n"] == 2 and result["messages"][-1].content == "done"
