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

## How failures are handled

| Kind | Example | Action |
|---|---|---|
| `transient` | 429, 503, timeout, dropped connection | Back off and retry the same call |
| `bad_output` | Malformed output, a call that breaks the schema (recognised by Jev, or raised as `BadOutputError`) | Retry with a hint saying what was wrong |
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

## Limitations

- **Budgets can overshoot by one call.** The check runs before a model call, but a call's cost is known only
  after it. keel has the same limitation.
- **No built-in prices.** You supply them; a model without a price, or a response without token usage, is an
  error rather than a silent $0. Dated names such as `gpt-4o-mini-2024-07-18` match the key `gpt-4o-mini`.
- **On resume after an escalation, the failing call runs once more** before the decision applies, because
  LangGraph re-runs the paused node. For a tool with side effects, make it idempotent or use `skip`.
- **Classification is only as good as the classifier.** Rules only recognise explicit signals (status codes,
  network errors, evenkeel's error classes): on keel's 75 error messages from public SDK issues they pick the
  right kind 29 times and the right action 39 times. Jev reads the message itself; see
  [keel's results](https://github.com/yodablocks/keel#failure-classification-rules-vs-jev) for its accuracy on
  real error messages, and `evals/` to measure it here.

## Develop

    uv sync --all-extras
    uv run pytest
