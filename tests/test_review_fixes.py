"""Regression tests for the findings of the final branch review."""

import asyncio
import socket

import pytest
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from langchain_evenkeel._core import decide
from langchain_evenkeel.budget import ModelPrice, PriceTable
from langchain_evenkeel.failure import (
    Fail,
    FailureContext,
    FailureVerdict,
    NeedsHumanError,
    RuleClassifier,
    default_policy,
)
from langchain_evenkeel.jev import JevClassifier
from langchain_evenkeel.middleware import EvenKeelMiddleware
from tests.fakes import ScriptedChatModel, ai, tool_call

PRICES = PriceTable({"fake-model": ModelPrice(input_per_mtok=1000, output_per_mtok=2000)})  # $0.20 a call


class RateLimited(Exception):
    status_code = 429


def agent_for(model, thread, tools=(), **kwargs):
    agent = create_agent(
        model, tools=list(tools), middleware=[EvenKeelMiddleware(**kwargs)], checkpointer=InMemorySaver()
    )
    return agent, {"configurable": {"thread_id": thread}}


def go():
    return {"messages": [HumanMessage("go")]}


# C1: a malformed resume value must not break the thread


def test_a_bad_escalation_answer_asks_again_and_a_good_one_then_works():
    model = ScriptedChatModel(
        script=[NeedsHumanError("x"), NeedsHumanError("x"), NeedsHumanError("x"), ai("done")]
    )
    agent, config = agent_for(model, "c1-model")
    agent.invoke(go(), config)
    again = agent.invoke(Command(resume="yes"), config)
    payload = again["__interrupt__"][0].value
    assert payload["type"] == "evenkeel_escalation" and "decision" in payload["error"]
    result = agent.invoke(Command(resume={"decision": "retry"}), config)
    assert result["messages"][-1].content == "done"


def test_a_bad_budget_answer_asks_again_and_a_good_one_then_works():
    @tool
    def search(q: str) -> str:
        """Search."""
        return "r"

    script = [
        ai(tool_calls=[tool_call("search", {"q": "a"}, "c1")]),
        ai(tool_calls=[tool_call("search", {"q": "b"}, "c2")]),
        ai("done"),
    ]
    agent, config = agent_for(
        ScriptedChatModel(script=script), "c1-budget", [search], prices=PRICES, budget_usd=0.3
    )
    agent.invoke(go(), config)
    again = agent.invoke(Command(resume={"approve": True}), config)
    assert "approve" in again["__interrupt__"][0].value["error"]
    result = agent.invoke(Command(resume={"approve": True, "budget_usd": 5}), config)
    assert result["messages"][-1].content == "done"


# C2: escalation must be bounded and must not replay forever


def test_a_tool_that_keeps_needing_a_person_fails_after_one_approved_retry():
    runs = {"n": 0}

    @tool
    def refund(order_id: str) -> str:
        """Refund an order."""
        runs["n"] += 1
        raise NeedsHumanError("refund over limit")

    model = ScriptedChatModel(script=[ai(tool_calls=[tool_call("refund", {"order_id": "42"})])])
    agent, config = agent_for(model, "c2-tool", [refund])
    agent.invoke(go(), config)
    with pytest.raises(NeedsHumanError):
        agent.invoke(Command(resume={"decision": "retry"}), config)
    assert runs["n"] <= 3


def test_a_reviewer_hint_on_a_tool_reaches_the_model():
    runs = {"n": 0}

    @tool
    def refund(order_id: str) -> str:
        """Refund an order."""
        runs["n"] += 1
        raise NeedsHumanError("refund over limit")

    model = ScriptedChatModel(script=[ai(tool_calls=[tool_call("refund", {"order_id": "42"})]), ai("done")])
    agent, config = agent_for(model, "c2-hint", [refund])
    agent.invoke(go(), config)
    result = agent.invoke(Command(resume={"decision": "retry", "hint": "approved by finance"}), config)
    sent_back = model.calls[-1][-1]
    assert isinstance(sent_back, ToolMessage) and "approved by finance" in sent_back.content
    assert result["messages"][-1].content == "done"
    assert runs["n"] == 2


def test_a_model_call_that_keeps_needing_a_person_fails_after_one_approved_retry():
    model = ScriptedChatModel(script=[NeedsHumanError("x"), NeedsHumanError("x"), NeedsHumanError("x")])
    agent, config = agent_for(model, "c2-model")
    agent.invoke(go(), config)
    with pytest.raises(NeedsHumanError):
        agent.invoke(Command(resume={"decision": "retry"}), config)
    assert len(model.calls) == 3


def test_escalations_count_against_the_attempt_limit():
    ctx = FailureContext(error=NeedsHumanError("x"), task="agent", attempt=3, max_attempts=3, step="model")
    decision = decide(ctx, FailureVerdict("needs_human", 1.0), default_policy())
    assert isinstance(decision.action, Fail)


# I3: spend must be counted when the model's reply is not the last message


def test_spend_counts_the_model_reply_even_when_another_message_follows():
    middleware = EvenKeelMiddleware(prices=PRICES)
    reply = ai(tool_calls=[tool_call("Answer", {"x": 1}, "s1")])
    state = {"messages": [HumanMessage("go"), reply, ToolMessage("ok", tool_call_id="s1")]}
    assert middleware.after_model(state, None) == {"evenkeel_spent_usd": pytest.approx(0.2)}


# I4: a failing classifier must not hide the user's error


class BrokenClassifier:
    def classify(self, ctx):
        raise RuntimeError("classifier is down")

    async def aclassify(self, ctx):
        raise RuntimeError("classifier is down")


def test_a_failing_classifier_falls_back_to_rules():
    model = ScriptedChatModel(script=[RateLimited("429"), ai("done")])
    agent, config = agent_for(model, "i4", classifier=BrokenClassifier(), sleep=lambda s: None)
    result = agent.invoke(go(), config)
    assert result["messages"][-1].content == "done"
    attempt = agent.get_state(config).values["evenkeel_attempts"][0]
    assert attempt["kind"] == "transient" and attempt["source"] == "rules (classifier error)"


def test_jev_with_only_an_async_client_falls_back_inside_a_running_loop():
    class AsyncOnly:
        async def system_one(self, **request):
            raise AssertionError("must not be called from sync code in a running loop")

    ctx = FailureContext(error=TypeError("bug"), task="t", attempt=1, max_attempts=3)

    async def classify_in_loop():
        return JevClassifier(async_client=AsyncOnly()).classify(ctx)

    verdict = asyncio.run(classify_in_loop())
    assert verdict.kind == "fatal" and verdict.source.startswith("rules")


# I5: errno collisions must not make local errors retryable


def test_a_missing_file_is_not_transient():
    error = FileNotFoundError(2, "No such file or directory")
    ctx = FailureContext(error=error, task="t", attempt=1, max_attempts=3)
    assert RuleClassifier().classify(ctx).kind == "fatal"


def test_temporary_dns_failure_is_still_transient():
    error = socket.gaierror(socket.EAI_AGAIN, "Temporary failure in name resolution")
    ctx = FailureContext(error=error, task="t", attempt=1, max_attempts=3)
    assert RuleClassifier().classify(ctx).kind == "transient"


# I6: the budget applies to each run, not to the whole thread


def test_each_run_on_a_thread_starts_with_zero_spend():
    model = ScriptedChatModel(script=[ai("first"), ai("second")])
    agent, config = agent_for(model, "i6-spend", prices=PRICES)
    agent.invoke(go(), config)
    agent.invoke(go(), config)
    assert agent.get_state(config).values["evenkeel_spent_usd"] == pytest.approx(0.2)


def test_a_new_run_is_not_blocked_by_the_previous_runs_budget():
    model = ScriptedChatModel(script=[ai("one"), ai("two"), ai("three")])
    agent, config = agent_for(model, "i6-budget", prices=PRICES, budget_usd=0.3)
    agent.invoke(go(), config)  # spends 0.2
    agent.invoke(go(), config)  # would be at 0.4 without the reset
    result = agent.invoke(go(), config)
    assert "__interrupt__" not in result and result["messages"][-1].content == "three"
    assert isinstance(result["messages"][-1], AIMessage)
