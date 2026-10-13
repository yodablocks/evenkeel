import pytest
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from langchain_evenkeel.failure import NeedsHumanError
from langchain_evenkeel.middleware import EvenKeelMiddleware
from tests.fakes import ScriptedChatModel, ai, tool_call


def start():
    return {"messages": [HumanMessage("refund order 42")]}


def test_needs_human_on_a_model_call_pauses_and_retry_resumes():
    # On resume LangGraph re-runs the model node, so the failing call runs once more
    # before the decision applies.
    model = ScriptedChatModel(
        script=[NeedsHumanError("refund over limit"), NeedsHumanError("refund over limit"), ai("done")]
    )
    agent = create_agent(model, middleware=[EvenKeelMiddleware()], checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "esc-model"}}
    paused = agent.invoke(start(), config)
    payload = paused["__interrupt__"][0].value
    assert payload["type"] == "evenkeel_escalation" and payload["kind"] == "needs_human"
    assert payload["accepted"] == ["retry", "fail"]
    result = agent.invoke(Command(resume={"decision": "retry", "hint": "approved by finance"}), config)
    assert result["messages"][-1].content == "done"
    assert "[evenkeel] approved by finance" in model.calls[-1][-1].content


def test_fail_decision_raises_the_original_error():
    model = ScriptedChatModel(
        script=[NeedsHumanError("refund over limit"), NeedsHumanError("refund over limit")]
    )
    agent = create_agent(model, middleware=[EvenKeelMiddleware()], checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "esc-fail"}}
    agent.invoke(start(), config)
    with pytest.raises(NeedsHumanError, match="refund over limit"):
        agent.invoke(Command(resume={"decision": "fail"}), config)


def test_skip_on_a_tool_returns_the_reviewers_message():
    @tool
    def refund(order_id: str) -> str:
        """Refund an order."""
        raise NeedsHumanError("refund of $900 is over the $500 limit")

    model = ScriptedChatModel(script=[ai(tool_calls=[tool_call("refund", {"order_id": "42"})]), ai("done")])
    agent = create_agent(
        model, tools=[refund], middleware=[EvenKeelMiddleware()], checkpointer=InMemorySaver()
    )
    config = {"configurable": {"thread_id": "esc-skip"}}
    paused = agent.invoke(start(), config)
    assert paused["__interrupt__"][0].value["accepted"] == ["retry", "fail", "skip"]
    result = agent.invoke(Command(resume={"decision": "skip", "message": "Refunded by hand."}), config)
    tool_message = next(m for m in result["messages"] if isinstance(m, ToolMessage))
    assert tool_message.content == "Refunded by hand."
    assert result["messages"][-1].content == "done"


def test_a_malformed_resume_value_is_a_clear_error():
    model = ScriptedChatModel(script=[NeedsHumanError("x"), NeedsHumanError("x")])
    agent = create_agent(model, middleware=[EvenKeelMiddleware()], checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "esc-bad"}}
    agent.invoke(start(), config)
    with pytest.raises(ValueError, match="decision"):
        agent.invoke(Command(resume="yes"), config)
