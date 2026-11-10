"""check_before_call=True: pause before a model call whose estimated worst case would cross the budget."""

import asyncio
import sys

import pytest
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from evenkeel.budget import ModelPrice, PriceTable, estimate_input_tokens, max_output_tokens
from evenkeel.failure import NeedsHumanError
from evenkeel.middleware import EvenKeelMiddleware
from tests.fakes import ScriptedChatModel, ai, tool_call

# $1 per million input tokens, $10 per million output tokens.
PRICES = PriceTable({"capped-model": ModelPrice(input_per_mtok=1.0, output_per_mtok=10.0)})


class CappedModel(ScriptedChatModel):
    """A scripted model that, like ChatAnthropic, says which model it is and caps its output."""

    model_name: str = "capped-model"
    max_tokens: int | None = 1000


@tool
def search(q: str) -> str:
    """Search."""
    return "result"


def reply(content: str = "", **kwargs) -> AIMessage:
    return ai(content, model="capped-model", tokens=(100, 50), **kwargs)


def build(script, thread, *, model_cls=CappedModel, model_kwargs=None, **middleware_kwargs):
    model = model_cls(script=script, **(model_kwargs or {}))
    middleware_kwargs.setdefault("prices", PRICES)
    agent = create_agent(
        model,
        tools=[search],
        middleware=[EvenKeelMiddleware(**middleware_kwargs)],
        checkpointer=InMemorySaver(),
    )
    return model, agent, {"configurable": {"thread_id": thread}}


def go():
    return {"messages": [HumanMessage("go")]}


# The estimate


def test_the_first_call_is_estimated_from_its_message_bytes():
    # '"hello"' is 7 bytes, plus 10 tokens of per-message overhead.
    assert estimate_input_tokens([HumanMessage("hello")]) == 17


def test_later_calls_start_from_the_last_reported_usage():
    messages = [HumanMessage("go"), reply(), ToolMessage("result", tool_call_id="c1")]
    # 100 in + 50 out from the last reply, then '"result"' (8 bytes) + 10 overhead for the tool result.
    assert estimate_input_tokens(messages) == 168


def test_max_output_tokens_prefers_the_call_settings_over_the_model():
    model = CappedModel(script=[], max_tokens=1000)
    assert max_output_tokens(model, {}) == 1000
    assert max_output_tokens(model, {"max_tokens": 200}) == 200
    assert max_output_tokens(CappedModel(script=[], max_tokens=None), {}) is None


# Off by default


def test_without_the_flag_a_call_runs_even_if_its_worst_case_would_cross_the_budget():
    # Worst case of the first call: 1000 output tokens at $10/M = $0.01, over a $0.005 budget.
    model, agent, config = build([reply("done")], "off", budget_usd=0.005)
    result = agent.invoke(go(), config)
    assert result["messages"][-1].content == "done"


def test_the_flag_needs_a_budget():
    with pytest.raises(ValueError, match="check_before_call needs budget_usd"):
        EvenKeelMiddleware(prices=PRICES, check_before_call=True)


# On


def test_a_call_whose_worst_case_crosses_the_budget_pauses_before_it_runs():
    model, agent, config = build([reply("done")], "pause", budget_usd=0.005, check_before_call=True)
    paused = agent.invoke(go(), config)
    payload = paused["__interrupt__"][0].value
    assert payload["type"] == "evenkeel_budget"
    assert payload["spent_usd"] == 0 and payload["budget_usd"] == 0.005
    assert payload["next_call_max_usd"] > 0.01  # $0.01 of output plus the input estimate
    assert model.calls == []


def test_a_call_that_fits_runs_without_a_pause():
    model, agent, config = build([reply("done")], "fits", budget_usd=1.0, check_before_call=True)
    result = agent.invoke(go(), config)
    assert "__interrupt__" not in result
    assert result["messages"][-1].content == "done"


def test_approving_enough_budget_runs_the_call_and_keeps_the_new_budget():
    model, agent, config = build([reply("done")], "approve", budget_usd=0.005, check_before_call=True)
    agent.invoke(go(), config)
    result = agent.invoke(Command(resume={"approve": True, "budget_usd": 1.0}), config)
    assert result["messages"][-1].content == "done"
    values = agent.get_state(config).values
    assert values["evenkeel_budget_usd"] == 1.0
    assert values["evenkeel_spent_usd"] == pytest.approx(0.0006)  # 100 in + 50 out at $1/$10 per million


def test_approving_too_little_asks_again_with_the_reason():
    model, agent, config = build([reply("done")], "too-little", budget_usd=0.005, check_before_call=True)
    agent.invoke(go(), config)
    again = agent.invoke(Command(resume={"approve": True, "budget_usd": 0.006}), config)
    payload = again["__interrupt__"][0].value
    assert "next call" in payload["error"]
    assert model.calls == []


def test_declining_ends_the_run_without_calling_the_model():
    model, agent, config = build([reply("done")], "decline", budget_usd=0.005, check_before_call=True)
    agent.invoke(go(), config)
    result = agent.invoke(Command(resume={"approve": False}), config)
    assert model.calls == []
    assert result["messages"][-1].content.startswith("Stopped by evenkeel")
    assert agent.get_state(config).values["evenkeel_spent_usd"] == 0


@pytest.mark.parametrize(
    "model_kwargs",
    [
        {"max_tokens": None},  # no output cap: no worst case
        {"model_name": "unpriced-model"},  # no price for the model the request names
    ],
)
def test_a_call_that_cannot_be_estimated_runs_as_before(model_kwargs):
    model, agent, config = build(
        [reply("done")], "no-estimate", model_kwargs=model_kwargs, budget_usd=0.005, check_before_call=True
    )
    result = agent.invoke(go(), config)
    assert result["messages"][-1].content == "done"


def test_the_estimate_grows_with_the_conversation_and_pauses_a_later_call():
    script = [reply(tool_calls=[tool_call("search", {"q": "a"}, "c1")]), reply("done")]
    # The first call's worst case (1000 tokens out = $0.01, plus a few hundred tokens in) fits in $0.0106.
    # After it, $0.0006 is spent, and the second call's worst case (168 tokens in + 1000 out = $0.010168) no
    # longer fits.
    model, agent, config = build(script, "grows", budget_usd=0.0106, check_before_call=True)
    paused = agent.invoke(go(), config)
    assert len(model.calls) == 1
    payload = paused["__interrupt__"][0].value
    assert payload["spent_usd"] == pytest.approx(0.0006)


def test_an_approval_then_an_escalation_in_the_same_call_both_resume_correctly():
    script = [NeedsHumanError("needs sign-off"), NeedsHumanError("needs sign-off"), reply("done")]
    model, agent, config = build(script, "both", budget_usd=0.005, check_before_call=True)
    first = agent.invoke(go(), config)
    assert first["__interrupt__"][0].value["type"] == "evenkeel_budget"
    second = agent.invoke(Command(resume={"approve": True, "budget_usd": 1.0}), config)
    assert second["__interrupt__"][0].value["type"] == "evenkeel_escalation"
    result = agent.invoke(Command(resume={"decision": "retry", "hint": "signed off"}), config)
    assert result["messages"][-1].content == "done"
    values = agent.get_state(config).values
    assert values["evenkeel_budget_usd"] == 1.0
    # LangGraph re-runs the call on resume, so only the final pass's failure is recorded, as without the flag.
    assert [(r["kind"], r["action"]) for r in values["evenkeel_attempts"]] == [("needs_human", "escalate")]


@pytest.mark.skipif(
    sys.version_info < (3, 11),
    reason="LangGraph's interrupt() cannot find its run inside async code before Python 3.11",
)
def test_the_async_path_pauses_and_resumes_the_same_way():
    async def run():
        model, agent, config = build([reply("done")], "async", budget_usd=0.005, check_before_call=True)
        paused = await agent.ainvoke(go(), config)
        assert paused["__interrupt__"][0].value["type"] == "evenkeel_budget"
        result = await agent.ainvoke(Command(resume={"approve": True, "budget_usd": 1.0}), config)
        assert result["messages"][-1].content == "done"
        assert (await agent.aget_state(config)).values["evenkeel_budget_usd"] == 1.0

    asyncio.run(run())
