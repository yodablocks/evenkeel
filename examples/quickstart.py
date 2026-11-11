"""evenkeel in five minutes, with no API key: a refund agent that stops before it can overspend."""

from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from pydantic import BaseModel

from evenkeel import EvenKeelMiddleware, ModelPrice, PriceTable


class Refund(BaseModel):
    order_id: str
    amount_usd: float


@tool
def lookup_order(order_id: str) -> str:
    """Look up an order."""
    return "A-1042: blender, $64, arrived broken"


def reply(name: str, **args) -> AIMessage:
    """A model reply calling `name`, with the token usage a provider reports."""
    usage = {"input_tokens": 20_000, "output_tokens": 500, "total_tokens": 20_500}
    call = {"name": name, "args": args, "id": f"call-{name}", "type": "tool_call"}
    return AIMessage(
        "", tool_calls=[call], usage_metadata=usage, response_metadata={"model_name": "gpt-4o-mini"}
    )


class ScriptedModel(GenericFakeChatModel):
    """Plays back replies. In your agent, use a real model with an output cap, such as
    init_chat_model("openai:gpt-4o-mini", max_tokens=1000)."""

    model_name: str = "gpt-4o-mini"
    max_tokens: int = 1000

    def bind_tools(self, tools, **kwargs):
        return self


model = ScriptedModel(
    messages=iter(
        [
            reply("lookup_order", order_id="A-1042"),
            reply("Refund", order_id="A-1042", amount_usd="sixty-four"),  # breaks the schema
            reply("Refund", order_id="A-1042", amount_usd=64),
        ]
    )
)
agent = create_agent(
    model,
    tools=[lookup_order],
    response_format=ToolStrategy(Refund, handle_errors=False),
    middleware=[
        EvenKeelMiddleware(
            budget_usd=0.005,
            prices=PriceTable({"gpt-4o-mini": ModelPrice(input_per_mtok=0.15, output_per_mtok=0.60)}),
            check_before_call=True,  # pause before a call whose worst case would cross the budget
        )
    ],
    checkpointer=InMemorySaver(),  # needed for pauses
)
config = {"configurable": {"thread_id": "quickstart"}}

# 1. One call ($0.0033) fits the $0.005 budget. The next could cost up to $0.0037, so evenkeel pauses first.
state = agent.invoke({"messages": [HumanMessage("Refund order A-1042, it arrived broken.")]}, config)
pause = state["__interrupt__"][0].value
spent, budget, next_max = pause["spent_usd"], pause["budget_usd"], pause["next_call_max_usd"]
print(f"Paused: spent ${spent:.4f} of ${budget}, next call up to ${next_max:.4f}")

# 2. A person raises the budget. The next reply breaks the Refund schema: evenkeel classifies it as bad output
#    and retries with a hint saying what was wrong. The rejected reply was paid for, so it counts.
state = agent.invoke(Command(resume={"approve": True, "budget_usd": 0.02}), config)
for attempt in state["evenkeel_attempts"]:
    print(f"Retried: {attempt['error_type']} -> {attempt['kind']}, {attempt['action']}")
print(f"Done: {state['structured_response']!r}, spent ${state['evenkeel_spent_usd']:.4f} for 3 calls")
