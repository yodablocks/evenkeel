"""evenkeel demo: one refund-support agent hits every kind of failure evenkeel handles, offline.

    uv run python examples/demo.py

The model is scripted and the tools misbehave on cue, so no API key is needed and every run prints the same
story. In one conversation:

1. Rate limit: the model API answers 429. evenkeel backs off and retries the same call.
2. Bad tool input: the model passes a malformed order id. The tool's error goes back to the model, which
   fixes it.
3. Bad output: the model's reply is unusable. evenkeel retries with a hint saying what was wrong.
4. Needs a person: a refund is over the agent's limit. The run pauses until a reviewer decides.
5. Over budget: the run has spent its budget. It pauses again, and the reviewer raises the budget.
"""

from __future__ import annotations

from typing import Any

from langchain.agents import create_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from pydantic import Field, PrivateAttr

from evenkeel import (
    BadInputError,
    BadOutputError,
    EvenKeelMiddleware,
    FailureContext,
    FailureVerdict,
    ModelPrice,
    NeedsHumanError,
    PriceTable,
    default_policy,
)

MODEL = "demo-model"
PRICES = PriceTable({MODEL: ModelPrice(input_per_mtok=3.0, output_per_mtok=15.0)})
BUDGET_USD = 0.03
REFUND_LIMIT_USD = 500
REQUEST = "My blender (order A-1042) arrived broken. I want a refund."


class RateLimitError(Exception):
    """What an SDK raises on HTTP 429."""

    status_code = 429


def _reply(content: str = "", tool_calls: list[dict[str, Any]] | None = None) -> AIMessage:
    # 2,000 tokens in and 300 out at the prices above: $0.0105 a call.
    return AIMessage(
        content=content,
        tool_calls=tool_calls or [],
        response_metadata={"model_name": MODEL},
        usage_metadata={"input_tokens": 2000, "output_tokens": 300, "total_tokens": 2300},
    )


def _call(name: str, args: dict[str, Any], call_id: str) -> dict[str, Any]:
    return {"name": name, "args": args, "id": call_id, "type": "tool_call"}


def _script() -> list[Any]:
    return [
        RateLimitError("429 Too Many Requests: rate limit exceeded"),
        _reply(tool_calls=[_call("lookup_order", {"order_id": "1042"}, "call_lookup_1")]),
        BadOutputError("The reply was cut off mid-sentence and has no answer for the customer"),
        _reply(tool_calls=[_call("lookup_order", {"order_id": "A-1042"}, "call_lookup_2")]),
        _reply(tool_calls=[_call("refund", {"order_id": "A-1042", "amount_usd": 640}, "call_refund")]),
        _reply("Your refund of $640 for order A-1042 is done. Sorry the blender arrived broken."),
    ]


class ScriptedModel(BaseChatModel):
    """Plays back a script: returns or raises the next item on each call, and says what it was sent."""

    script: list[Any] = Field(default_factory=_script)
    _log: list[str] = PrivateAttr(default_factory=list)  # private, so pydantic shares the list, not a copy
    calls: int = 0

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools: Any, **kwargs: Any) -> ScriptedModel:
        return self

    def _generate(
        self, messages: list[BaseMessage], stop: Any = None, run_manager: Any = None, **kwargs: Any
    ) -> ChatResult:
        self.calls += 1
        last = messages[-1]
        if isinstance(last, HumanMessage) and str(last.content).startswith("[evenkeel]"):
            say(self._log, "model", f"call {self.calls}, with evenkeel's hint: {last.content}")
        else:
            say(self._log, "model", f"call {self.calls}")
        item = self.script.pop(0)
        if isinstance(item, BaseException):
            raise item
        return ChatResult(generations=[ChatGeneration(message=item)])


def say(log: list[str], who: str, text: str) -> None:
    line = f"[{who}] {text}"
    log.append(line)
    print(line)


def build_agent(log: list[str], resuming: dict[str, bool]) -> tuple[Any, ScriptedModel]:
    @tool
    def lookup_order(order_id: str) -> str:
        """Look up an order by id, like A-1042."""
        if not (order_id.startswith("A-") and order_id[2:].isdigit()):
            say(log, "tool", f"lookup_order({order_id!r}) rejected the id")
            raise BadInputError(f"Order id {order_id!r} is malformed: ids look like A-1042")
        say(log, "tool", f"lookup_order({order_id!r}) found a $640 blender, delivered broken")
        return '{"order_id": "A-1042", "item": "blender", "paid_usd": 640, "status": "delivered, damaged"}'

    @tool
    def refund(order_id: str, amount_usd: float) -> str:
        """Refund an order."""
        if amount_usd > REFUND_LIMIT_USD:
            say(log, "tool", f"refund(${amount_usd:.0f}) is over the ${REFUND_LIMIT_USD} limit")
            raise NeedsHumanError(f"A refund of ${amount_usd:.0f} is over the ${REFUND_LIMIT_USD} limit")
        return f"Refunded ${amount_usd:.0f} on {order_id}"

    base = default_policy(base_s=0.2, random=lambda: 0.5)  # short, fixed backoff so the demo is repeatable

    def narrated(verdict: FailureVerdict, ctx: FailureContext):
        action = base(verdict, ctx)
        if action.type == "escalate" and resuming["on"]:
            resuming["on"] = False
            say(log, "evenkeel", "LangGraph re-ran the paused call on resume. The decision applies now.")
            return action
        if verdict.kind == "bad_input" and (ctx.step or "").startswith("tool:"):
            detail = "hand the error back to the model to fix its arguments"
        else:
            detail = {
                "retry": lambda: f"back off {getattr(action, 'delay_s', 0):.1f}s and retry",
                "retry_modified": lambda: "retry with a hint",
                "escalate": lambda: "pause for a person",
                "fail": lambda: "fail",
            }[action.type]()
        say(
            log,
            "evenkeel",
            f"{ctx.step} failed: {ctx.error}\n"
            f"           -> {verdict.kind} (confidence {verdict.confidence:.2f}), so {detail}",
        )
        return action

    model = ScriptedModel()
    model._log = log
    agent = create_agent(
        model,
        tools=[lookup_order, refund],
        system_prompt="You are a support agent. Refunds over $500 need a person.",
        middleware=[EvenKeelMiddleware(policy=narrated, budget_usd=BUDGET_USD, prices=PRICES)],
        checkpointer=InMemorySaver(),
    )
    return agent, model


def main() -> dict[str, Any]:
    log: list[str] = []
    # Set while a reviewer's decision is being applied, to narrate LangGraph's replay of the paused call.
    resuming = {"on": False}
    agent, _ = build_agent(log, resuming)
    config = {"configurable": {"thread_id": "demo"}}

    say(log, "customer", REQUEST)
    state = agent.invoke({"messages": [HumanMessage(REQUEST)]}, config)

    answers = {
        "evenkeel_escalation": {"decision": "skip", "message": "Refunded $640 by hand, ticket FIN-88."},
        "evenkeel_budget": {"approve": True, "budget_usd": 0.05},
    }
    while "__interrupt__" in state:
        payload = state["__interrupt__"][0].value
        if payload["type"] == "evenkeel_escalation":
            say(log, "reviewer", f"paused: {payload['error']}. Decision: refund by hand and skip the tool.")
            resuming["on"] = True
        else:
            say(
                log,
                "reviewer",
                f"paused: spent ${payload['spent_usd']:.4f} of a ${payload['budget_usd']:.2f} budget. "
                "Decision: raise it to $0.05.",
            )
        state = agent.invoke(Command(resume=answers[payload["type"]]), config)

    say(log, "agent", str(state["messages"][-1].content))
    say(log, "demo", f"spent ${state['evenkeel_spent_usd']:.4f}; every failure evenkeel handled:")
    for record in state.get("evenkeel_attempts", []):
        line = f"  {record['call']}: {record['error_type']} -> {record['kind']}, {record['action']}"
        log.append(line)
        print(line)
    return {"log": log, "state": state}


if __name__ == "__main__":
    main()
