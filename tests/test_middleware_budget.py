import pytest
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from langchain_evenkeel.budget import MissingUsageError, ModelPrice, PriceTable, UnknownModelPriceError
from langchain_evenkeel.middleware import EvenKeelMiddleware
from tests.fakes import ScriptedChatModel, ai, tool_call

# 100 input + 50 output tokens at $1000/$2000 per million = $0.20 per model call.
PRICES = PriceTable({"fake-model": ModelPrice(input_per_mtok=1000, output_per_mtok=2000)})


@tool
def search(q: str) -> str:
    """Search."""
    return "result"


def three_turn_script():
    return [
        ai(tool_calls=[tool_call("search", {"q": "a"}, "c1")]),
        ai(tool_calls=[tool_call("search", {"q": "b"}, "c2")]),
        ai("done"),
    ]


def build(script, thread, **kwargs):
    model = ScriptedChatModel(script=script)
    agent = create_agent(
        model, tools=[search], middleware=[EvenKeelMiddleware(**kwargs)], checkpointer=InMemorySaver()
    )
    return model, agent, {"configurable": {"thread_id": thread}}


def test_spend_is_tracked_without_a_budget():
    model, agent, config = build(three_turn_script(), "spend", prices=PRICES)
    agent.invoke({"messages": [HumanMessage("go")]}, config)
    assert agent.get_state(config).values["evenkeel_spent_usd"] == pytest.approx(0.6)


def test_over_budget_pauses_before_the_next_model_call():
    model, agent, config = build(three_turn_script(), "pause", prices=PRICES, budget_usd=0.3)
    paused = agent.invoke({"messages": [HumanMessage("go")]}, config)
    payload = paused["__interrupt__"][0].value
    assert payload["type"] == "evenkeel_budget"
    assert payload["spent_usd"] == pytest.approx(0.4) and payload["budget_usd"] == 0.3
    assert len(model.calls) == 2


def test_approval_raises_the_budget_and_finishes():
    model, agent, config = build(three_turn_script(), "approve", prices=PRICES, budget_usd=0.3)
    agent.invoke({"messages": [HumanMessage("go")]}, config)
    result = agent.invoke(Command(resume={"approve": True, "budget_usd": 1.0}), config)
    assert result["messages"][-1].content == "done"
    values = agent.get_state(config).values
    assert values["evenkeel_budget_usd"] == 1.0 and values["evenkeel_spent_usd"] == pytest.approx(0.6)


def test_rejection_ends_the_run_cleanly():
    model, agent, config = build(three_turn_script(), "reject", prices=PRICES, budget_usd=0.3)
    agent.invoke({"messages": [HumanMessage("go")]}, config)
    result = agent.invoke(Command(resume={"approve": False}), config)
    last = result["messages"][-1]
    assert isinstance(last, AIMessage) and last.content.startswith("Stopped by evenkeel:")
    assert len(model.calls) == 2


def test_unpriced_model_is_an_error():
    model, agent, config = build([ai("done", model="other-model")], "unpriced", prices=PRICES, budget_usd=1.0)
    with pytest.raises(UnknownModelPriceError, match="other-model"):
        agent.invoke({"messages": [HumanMessage("go")]}, config)


def test_missing_usage_with_a_budget_is_an_error():
    message = AIMessage(content="done", response_metadata={"model_name": "fake-model"})
    model, agent, config = build([message], "no-usage", prices=PRICES, budget_usd=1.0)
    with pytest.raises(MissingUsageError):
        agent.invoke({"messages": [HumanMessage("go")]}, config)
