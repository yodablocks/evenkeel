# evenkeel

Keeps a LangChain agent on an even keel when things fail: failures are classified and handled by kind,
bad output is retried with a hint, spend is tracked in dollars with a budget, and failures that need a person
pause the run for approval.

Ported from [keel](https://github.com/yodablocks/keel), a durable execution engine for AI agents.
An independent project, not affiliated with LangChain. Experimental: 0.x, expect changes.

## What it adds to LangChain's built-in middleware

| Need | Built in | EvenKeelMiddleware |
|---|---|---|
| Retries | `ModelRetryMiddleware`, `ToolRetryMiddleware` retry with the same input | Classifies the failure first: back off, retry **with a hint**, fail at once, or escalate |
| Limits | `ModelCallLimitMiddleware` counts calls | Tracks **dollars** from token usage and pauses at a budget |
| Approvals | `HumanInTheLoopMiddleware` asks before chosen tools | Escalates **when a failure needs a person**, from any model or tool call |

Use it alongside the built-ins: `ModelFallbackMiddleware` for switching models, `HumanInTheLoopMiddleware`
for approvals fixed in advance.

## Install

    pip install evenkeel          # rule-based classification
    pip install "evenkeel[jev]"   # adds TypeSafe's Jev classifier

## Try it in five minutes

No API key needed: the model is scripted, and everything else is real LangChain and evenkeel.

    pip install evenkeel

Save this as `quickstart.py` and run `python quickstart.py`:

```python
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
```

It prints:

```
Paused: spent $0.0033 of $0.005, next call up to $0.0037
Retried: StructuredOutputValidationError -> bad_output, retry_modified
Done: Refund(order_id='A-1042', amount_usd=64.0), spent $0.0099 for 3 calls
```

1. **It stops before it can overspend.** One call fits the $0.005 budget. The next could cost up to $0.0037,
   so evenkeel pauses before making it, and a person decides.
2. **It retries bad output with a hint.** The reply breaks the `Refund` schema, so evenkeel retries it with a
   hint saying what was wrong. The rejected reply was paid for, so it counts toward the spend.

On your own agent, replace the scripted model with a real one that caps its output (the cap is how evenkeel
works out a call's worst case), and add a price for it:

```python
from langchain.chat_models import init_chat_model

model = init_chat_model("openai:gpt-4o-mini", max_tokens=1000)
```

Tried it? [Open an issue](https://github.com/yodablocks/evenkeel/issues/new) and say which part you used, the
budget or the failure handling, and what got in the way. It is the most useful thing you can send.

## Use

```python
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from evenkeel import EvenKeelMiddleware, ModelPrice, PriceTable

agent = create_agent(
    "openai:gpt-4o-mini",
    tools=[...],
    middleware=[
        EvenKeelMiddleware(
            budget_usd=0.50,
            prices=PriceTable({"gpt-4o-mini": ModelPrice(input_per_mtok=0.15, output_per_mtok=0.60)}),
        )
    ],
    checkpointer=InMemorySaver(),  # needed for pauses
)
```

With Jev (needs `TYPESAFE_API_KEY`):

```python
from typesafe_sdk import TypeSafeClient
from evenkeel import EvenKeelMiddleware, JevClassifier

EvenKeelMiddleware(classifier=JevClassifier(TypeSafeClient()))
```

## Demo

From a clone of the repo, `uv run python examples/demo.py` runs one refund-support agent through every kind of
failure evenkeel handles.
It's offline: the model is scripted and the tools fail on cue, so no API key is needed. In one conversation:

1. **Rate limit.** The model API answers 429. evenkeel backs off and retries the same call.
2. **Bad tool input.** The model passes a malformed order id. The tool's error goes back to the model, which
   fixes it.
3. **Bad output.** The model's reply is unusable. evenkeel retries with a hint saying what was wrong.
4. **Budget.** With $0.021 of its $0.03 budget spent, the next call could cost up to $0.0117. evenkeel pauses
   before making it (`check_before_call=True`), so the budget is never overshot, and the reviewer raises it.
5. **Needs a person.** A $640 refund is over the agent's $500 limit. The run pauses, and a reviewer refunds by
   hand and skips the tool.

```
[customer] My blender (order A-1042) arrived broken. I want a refund.
[model] call 1
[evenkeel] model failed: 429 Too Many Requests: rate limit exceeded
           -> transient (confidence 0.90), so back off 0.1s and retry
[model] call 2
[tool] lookup_order('1042') rejected the id
[evenkeel] tool:lookup_order failed: Order id '1042' is malformed: ids look like A-1042
           -> bad_input (confidence 1.00), so hand the error back to the model to fix its arguments
[model] call 3
[evenkeel] model failed: The reply was cut off mid-sentence and has no answer for the customer
           -> bad_output (confidence 1.00), so retry with a hint
[model] call 4, with evenkeel's hint: [evenkeel] The previous attempt produced unusable output: The reply was cut off mid-sentence and has no answer for the customer
[tool] lookup_order('A-1042') found a $640 blender, delivered broken
[reviewer] paused before a call: spent $0.0210 of a $0.03 budget, and the next call could cost up to $0.0117. Decision: raise it to $0.05.
[model] call 5
[tool] refund($640) is over the $500 limit
[evenkeel] tool:refund failed: A refund of $640 is over the $500 limit
           -> needs_human (confidence 1.00), so pause for a person
[reviewer] paused: A refund of $640 is over the $500 limit. Decision: refund by hand and skip the tool.
[tool] refund($640) is over the $500 limit
[evenkeel] LangGraph re-ran the paused call on resume. The decision applies now.
[model] call 6
[agent] Your refund of $640 for order A-1042 is done. Sorry the blender arrived broken.
[demo] spent $0.0420; every failure evenkeel handled:
  model: RateLimitError -> transient, retry
  tool:lookup_order: BadInputError -> bad_input, returned_to_model
  model: BadOutputError -> bad_output, retry_modified
  tool:refund: NeedsHumanError -> needs_human, skipped_by_reviewer
```

`tests/test_demo.py` runs it and checks every moment.

## How failures are handled

| Kind | Example | Action |
|---|---|---|
| `transient` | 429, 503, timeout, dropped connection | Back off and retry the same call |
| `bad_output` | Malformed output, a call that breaks the schema (`BadOutputError`, LangChain's structured output and output parser errors, or recognised by Jev) | Retry with a hint saying what was wrong |
| `bad_input` | Input a tool rejects (`BadInputError`, HTTP 400 or 422, a pydantic `ValidationError`) | Model call: fail. Tool call: return the error to the model so it fixes its arguments |
| `needs_human` | Refund over a limit, a policy block | Pause with `interrupt()` and wait for a decision |
| `fatal` | A bug, bad credentials | Re-raise the original error at once |

Retries and escalations stop at `max_attempts` (default 4). Each failure is recorded (error, kind, confidence,
which classifier decided, action): in the agent's state under `evenkeel_attempts` when the call ends in a
result, and on the raised error as `error.evenkeel_attempts` when it finally fails.

Some failures never reach the middleware, because LangChain turns them into messages for the model itself:
a call to a tool that does not exist, and tool arguments that fail the tool's schema.

## Pauses and how to resume

A failure that needs a person:

```python
from langgraph.types import Command

agent.invoke(Command(resume={"decision": "retry", "hint": "approved by finance"}), config)
agent.invoke(Command(resume={"decision": "fail"}), config)
agent.invoke(Command(resume={"decision": "skip", "message": "Refunded by hand."}), config)  # tool calls only
```

A `retry` with a `hint` on a tool call sends the reviewer's words to the model, which can call the tool again; a
tool cannot take a hint itself. A call that still needs a person after one approved retry fails. An answer
that is not one of these values does not raise: the run pauses again, with an `error` field in the payload
saying what was wrong.

Over budget:

```python
agent.invoke(Command(resume={"approve": True, "budget_usd": 1.00}), config)
agent.invoke(Command(resume={"approve": False}), config)  # ends the run
```

The budget applies to each run: every new `invoke` on a thread starts with zero spend and the configured
budget.

A reply that was paid for and then rejected counts too, so retries can't run up cost unseen. evenkeel adds its
cost when the error carries the reply: LangChain's structured output errors do (`ai_message`), and so does a
`BadOutputError` raised with the reply as `output`:

```python
raise BadOutputError("The reply has no answer for the customer", output=reply)  # reply: the AIMessage
```

An error without the reply adds nothing, as before. Rejected replies are added when the call finally succeeds,
so they are not counted when the call instead ends with an error, or pauses for a person (LangGraph replays
the call on resume, and counts the replay's replies). The budget sees them before the next model call, not
between retries of the same call; `max_attempts` bounds those.

### Checking a call before it runs

By default the budget is checked against what has been spent, so the call that crosses it still runs. With
`check_before_call=True`, evenkeel also estimates the most the next call can cost and pauses before it when
that would cross the budget:

```python
EvenKeelMiddleware(budget_usd=0.50, prices=prices, check_before_call=True)
```

The estimate is the call's output cap (`max_tokens`, `max_completion_tokens` or `max_output_tokens`, from the
call's settings or the model) at the output price, plus its input at the input price. Input is counted from
the last reply's reported usage, plus the new messages at one token per UTF-8 byte, which overcounts. The pause
payload adds `next_call_max_usd`, and an approval must raise the budget above `spent_usd` plus that amount.

A call is not checked, and runs as before, when its model has no output cap (ChatOpenAI by default), when the
model's name isn't readable, or when that name has no price. evenkeel then emits an `EvenKeelWarning` once
for each reason, so the check is never off without you knowing. For a model with no cap, you can give one:

```python
EvenKeelMiddleware(budget_usd=0.50, prices=prices, check_before_call=True, assumed_max_output_tokens=4096)
```

That number is an assumption, not a limit: evenkeel doesn't pass it to the model. Prefer setting `max_tokens` on
the model, which the provider enforces; the model's own cap always wins over the assumed one.

## Limitations

- **Budgets can overshoot by one call.** The check runs before a model call, but a call's cost is known only
  after it. With `check_before_call=True` and an output cap on the model, the
  overshoot is limited to the estimate's error: provider overhead for tool formatting, and other middleware
  changing the model, prompt or tools after evenkeel has looked at the call.
- **No built-in prices.** You supply them; a model without a price, or a response without token usage, is an
  error rather than a silent $0. Dated or versioned names (`gpt-4o-mini-2024-07-18`, `claude-3-5-sonnet-20241022`, `gpt-4-0613`, `...-latest`)
  use the price of their base name; any other variant (`gpt-4o-mini` against `gpt-4`, `gpt-4-turbo`) needs its own
  entry.
- **On resume after an escalation, the failing call runs once more** before the decision applies, because
  LangGraph re-runs the paused node. For a tool with side effects, make it idempotent or use `skip`.
- **Classification is only as good as the classifier.** Rules only recognise explicit signals (status codes,
  network errors, evenkeel's error classes): on keel's 75 error messages from public SDK issues they pick the
  right kind 29 times and the right action 39 times. With Jev the results are below.

## Classifier accuracy

Measured with `evals/jev_eval.py` on keel's labelled error sets: 75 error messages from public SDK issues and
60 sampled from 13 other repositories. Jev pinned to `jev-1.13.0`, one run on 2026-10-14. "Right action" counts
a verdict as right when the default policy does the same thing as for the label (`bad_input` and `fatal` both
fail).

| Classifier | Public set, right kind | Public set, right action | Fresh set, right kind | Fresh set, right action |
|---|---|---|---|---|
| Rules only | 29 / 75 (39%) | 39 / 75 (52%) | 43 / 60 (72%) | 48 / 60 (80%) |
| `JevClassifier` (rules for explicit signals, Jev for the rest, rules below 0.5 confidence) | 50 / 75 (67%) | 63 / 75 (84%) | 49 / 60 (82%) | 56 / 60 (93%) |

Caveats: the labels are one person's (keel's author, who also wrote the classifier prompt), most messages come
from development rather than production, and Jev's answers vary by about one case between runs. On the public
set Jev answered 66 cases and the rules 9 (Jev below 0.5 confidence), of which 7 were wrong. keel's TypeScript
cascade scored 46 to 47 and 59 to 60 on the same set ([its results](https://github.com/yodablocks/keel#failure-classification-rules-vs-jev));
the difference is in how often Jev falls below the confidence floor, and its cause has not been established.

## Develop

    uv sync --all-extras
    uv run pytest
