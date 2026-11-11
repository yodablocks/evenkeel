"""A reply that was billed and then rejected counts toward spend, so retries can't run up cost unseen."""

import asyncio

import pytest
from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware
from langchain.agents.structured_output import MultipleStructuredOutputsError, StructuredOutputValidationError
from langchain_core.exceptions import OutputParserException
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

from evenkeel.budget import ModelPrice, PriceTable, UnknownModelPriceError
from evenkeel.failure import BadOutputError, FailureContext, RuleClassifier
from evenkeel.middleware import EvenKeelMiddleware, EvenKeelWarning
from tests.fakes import ScriptedChatModel, ai, tool_call

# $1 per million tokens in and out, so 100 + 50 tokens cost $0.00015.
PRICES = PriceTable({"priced-model": ModelPrice(input_per_mtok=1.0, output_per_mtok=1.0)})
ONE_CALL = 0.00015


def reply(content: str) -> AIMessage:
    return ai(content, model="priced-model", tokens=(100, 50))


class RejectFirstReplies(AgentMiddleware):
    """Like an output validator: runs inside evenkeel, after the model has replied (and been paid)."""

    def __init__(self, rejections: int, make_error):
        super().__init__()
        self.left = rejections
        self.make_error = make_error

    def wrap_model_call(self, request, handler):
        response = handler(request)
        if self.left:
            self.left -= 1
            raise self.make_error(response.result[-1])
        return response

    async def awrap_model_call(self, request, handler):
        response = await handler(request)
        if self.left:
            self.left -= 1
            raise self.make_error(response.result[-1])
        return response


def run(make_error, *, rejections=2, prices=PRICES, **kwargs):
    model = ScriptedChatModel(script=[reply("bad"), reply("bad"), reply("good")])
    agent = create_agent(
        model,
        middleware=[EvenKeelMiddleware(prices=prices, **kwargs), RejectFirstReplies(rejections, make_error)],
        checkpointer=InMemorySaver(),
    )
    config = {"configurable": {"thread_id": "t"}}
    return model, agent, agent.invoke({"messages": [HumanMessage("go")]}, config)


def test_rejected_replies_that_carry_their_message_are_counted():
    model, _, result = run(lambda message: BadOutputError("unusable", output=message))
    assert len(model.calls) == 3
    assert result["messages"][-1].content == "good"
    assert result["evenkeel_spent_usd"] == pytest.approx(3 * ONE_CALL), (
        "two rejected replies plus the good one"
    )


def test_langchain_structured_output_errors_are_counted_through_their_ai_message():
    def error(message):
        return StructuredOutputValidationError("Answer", ValueError("missing field"), message)

    _, _, result = run(error)
    assert result["evenkeel_spent_usd"] == pytest.approx(3 * ONE_CALL)


def test_a_failure_with_no_reply_attached_costs_nothing_as_before():
    _, _, result = run(lambda message: BadOutputError("unusable"))
    assert result["evenkeel_spent_usd"] == pytest.approx(ONE_CALL)


def test_a_rejected_reply_with_no_price_warns_instead_of_hiding_the_error():
    unpriced = PriceTable({"other-model": ModelPrice(input_per_mtok=1.0, output_per_mtok=1.0)})
    with pytest.warns(EvenKeelWarning, match="rejected reply"), pytest.raises(UnknownModelPriceError):
        # The good reply is unpriced too, so the run itself still fails on the missing price, as before.
        run(lambda message: BadOutputError("unusable", output=message), rejections=1, prices=unpriced)


@pytest.mark.parametrize(
    "error",
    [
        StructuredOutputValidationError("Answer", ValueError("missing field"), AIMessage("")),
        MultipleStructuredOutputsError(["A", "B"], AIMessage("")),
        OutputParserException("not JSON"),
    ],
)
def test_langchain_output_errors_are_bad_output_so_the_retry_gets_a_hint(error):
    ctx = FailureContext(error=error, task="agent", attempt=1, max_attempts=4, step="model")
    verdict = RuleClassifier().classify(ctx)
    assert verdict.kind == "bad_output"


def test_the_async_path_counts_rejected_replies_too():
    async def go():
        model = ScriptedChatModel(script=[reply("bad"), reply("good")])
        agent = create_agent(
            model,
            middleware=[
                EvenKeelMiddleware(prices=PRICES),
                RejectFirstReplies(1, lambda message: BadOutputError("unusable", output=message)),
            ],
            checkpointer=InMemorySaver(),
        )
        return await agent.ainvoke({"messages": [HumanMessage("go")]}, {"configurable": {"thread_id": "a"}})

    result = asyncio.run(go())
    assert result["evenkeel_spent_usd"] == pytest.approx(2 * ONE_CALL)


@tool
def search(q: str) -> str:
    """Search."""
    return "result"


def test_a_rejected_reply_can_push_the_run_over_budget_so_the_next_call_pauses():
    # One call costs $0.00015. The accepted reply alone fits a $0.0002 budget; with the rejected one, $0.0003
    # is spent and the next model call pauses for a budget decision instead of running.
    script = [
        reply("bad"),
        ai(tool_calls=[tool_call("search", {"q": "a"})], model="priced-model"),
        reply("done"),
    ]
    model = ScriptedChatModel(script=script)
    agent = create_agent(
        model,
        tools=[search],
        middleware=[
            EvenKeelMiddleware(prices=PRICES, budget_usd=0.0002),
            RejectFirstReplies(1, lambda message: BadOutputError("unusable", output=message)),
        ],
        checkpointer=InMemorySaver(),
    )
    paused = agent.invoke({"messages": [HumanMessage("go")]}, {"configurable": {"thread_id": "budget"}})

    payload = paused["__interrupt__"][0].value
    assert payload["type"] == "evenkeel_budget"
    assert payload["spent_usd"] == pytest.approx(2 * ONE_CALL)
    assert len(model.calls) == 2, "the third call never ran"
