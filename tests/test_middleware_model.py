import asyncio

import pytest
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from langchain_evenkeel._compat import AgentMiddleware, interrupt
from langchain_evenkeel.failure import FailureVerdict
from langchain_evenkeel.middleware import EvenKeelMiddleware
from tests.fakes import ScriptedChatModel, ai


class RateLimited(Exception):
    status_code = 429


class Unavailable(Exception):
    status_code = 503


class StubClassifier:
    """Answers bad_output for one message, like Jev recognising a hallucinated tool name."""

    def classify(self, ctx):
        if "not in the provided tool list" in str(ctx.error):
            return FailureVerdict("bad_output", 0.87, "jev")
        return FailureVerdict("fatal", 0.5)

    async def aclassify(self, ctx):
        return self.classify(ctx)


def run(model, middleware, thread="t"):
    agent = create_agent(model, middleware=[middleware], checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": thread}}
    result = agent.invoke({"messages": [HumanMessage("hi")]}, config)
    return agent, config, result


def test_transient_errors_back_off_and_retry():
    slept: list[float] = []
    model = ScriptedChatModel(script=[RateLimited("429"), ai("done")])
    agent, config, result = run(model, EvenKeelMiddleware(sleep=slept.append))
    assert result["messages"][-1].content == "done"
    assert len(model.calls) == 2 and len(slept) == 1
    attempts = agent.get_state(config).values["evenkeel_attempts"]
    assert [(a["kind"], a["action"]) for a in attempts] == [("transient", "retry")]


def test_bad_output_retries_with_a_hint():
    error = Exception("Model requested tool `serch_web`, which is not in the provided tool list")
    model = ScriptedChatModel(script=[error, ai("done")])
    run(model, EvenKeelMiddleware(classifier=StubClassifier()))
    hint = model.calls[1][-1]
    assert isinstance(hint, HumanMessage) and hint.content.startswith("[evenkeel] ")
    assert "serch_web" in hint.content
    assert not any("[evenkeel]" in str(m.content) for m in model.calls[0])


def test_fatal_errors_are_re_raised_unchanged_with_records_attached():
    bug = TypeError("'NoneType' object is not subscriptable")
    model = ScriptedChatModel(script=[bug])
    with pytest.raises(TypeError) as raised:
        run(model, EvenKeelMiddleware())
    assert raised.value is bug
    assert raised.value.evenkeel_attempts[0]["kind"] == "fatal"
    assert len(model.calls) == 1


def test_a_failure_that_never_goes_away_stops_at_max_attempts():
    model = ScriptedChatModel(script=[Unavailable("503") for _ in range(5)])
    with pytest.raises(Unavailable) as raised:
        run(model, EvenKeelMiddleware(max_attempts=3, sleep=lambda s: None))
    assert len(model.calls) == 3
    assert [a["action"] for a in raised.value.evenkeel_attempts] == ["retry", "retry", "fail"]


def test_langgraph_interrupts_pass_through_untouched():
    class Approver(AgentMiddleware):
        def wrap_model_call(self, request, handler):
            interrupt("approve?")
            return handler(request)

    model = ScriptedChatModel(script=[ai("done")])
    agent = create_agent(model, middleware=[EvenKeelMiddleware(), Approver()], checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "pass"}}
    result = agent.invoke({"messages": [HumanMessage("hi")]}, config)
    assert result["__interrupt__"][0].value == "approve?"
    assert agent.get_state(config).values.get("evenkeel_attempts", []) == []


def test_async_agents_get_the_same_retry():
    slept: list[float] = []

    async def asleep(seconds: float) -> None:
        slept.append(seconds)

    model = ScriptedChatModel(script=[RateLimited("429"), ai("done")])
    agent = create_agent(model, middleware=[EvenKeelMiddleware(asleep=asleep)], checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "async"}}
    result = asyncio.run(agent.ainvoke({"messages": [HumanMessage("hi")]}, config))
    assert result["messages"][-1].content == "done"
    assert len(model.calls) == 2 and len(slept) == 1


def test_constructor_validates_its_arguments():
    with pytest.raises(ValueError, match="prices"):
        EvenKeelMiddleware(budget_usd=1.0)
    with pytest.raises(ValueError, match="max_attempts"):
        EvenKeelMiddleware(max_attempts=0)
    with pytest.raises(ValueError, match="budget_usd"):
        from langchain_evenkeel.budget import ModelPrice, PriceTable

        EvenKeelMiddleware(budget_usd=0, prices=PriceTable({"m": ModelPrice(1, 1)}))
