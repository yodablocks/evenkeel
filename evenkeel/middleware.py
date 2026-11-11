"""EvenKeelMiddleware: classified retries with hints, dollar budgets and escalation for LangChain agents."""

from __future__ import annotations

import asyncio
import json
import operator
import time
import warnings
from collections.abc import Awaitable, Callable
from dataclasses import replace
from typing import Annotated, Any, NotRequired

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.utils.function_calling import convert_to_openai_tool

from ._compat import (
    AgentMiddleware,
    AgentState,
    Command,
    ExtendedModelResponse,
    GraphBubbleUp,
    ModelRequest,
    ModelResponse,
    ToolCallRequest,
    hook_config,
    interrupt,
)
from ._core import (
    AttemptRecord,
    Decision,
    attach_attempts,
    decide,
    escalation_payload,
    parse_budget_resume,
    parse_resume,
)
from .budget import (
    PriceTable,
    UnknownModelPriceError,
    chat_model_name,
    estimate_input_tokens,
    max_output_tokens,
)
from .failure import (
    Classifier,
    Escalate,
    FailureContext,
    FailurePolicy,
    FailureVerdict,
    Retry,
    RetryWithHint,
    RuleClassifier,
    default_policy,
    error_message,
)


class EvenKeelState(AgentState):
    evenkeel_spent_usd: NotRequired[float]
    evenkeel_budget_usd: NotRequired[float]
    evenkeel_attempts: NotRequired[Annotated[list[AttemptRecord], operator.add]]


class _RetryRequest:
    """Internal signal from the decision helpers: try the call again with this request."""

    def __init__(self, request: Any, delay_s: float = 0.0, *, escalated: bool = False) -> None:
        self.request = request
        self.delay_s = delay_s
        self.escalated = escalated


def _ask(payload: dict[str, Any], parse: Callable[[Any], Any]) -> Any:
    """Pause for a person and parse the answer. A malformed answer asks again with the error instead of
    raising: LangGraph keeps a resume value once given, so raising would replay it on every later resume."""
    while True:
        answer = interrupt(payload)
        try:
            return parse(answer)
        except ValueError as error:
            payload = {**payload, "error": str(error)}


def _with_hint(request: ModelRequest, hint: str) -> ModelRequest:
    return request.override(messages=[*request.messages, HumanMessage(f"[evenkeel] {hint}")])


class EvenKeelWarning(UserWarning):
    """evenkeel was asked to do something it cannot do for this run, such as check a call it cannot price."""


# Marks evenkeel's own messages, which no model produced, so spend tracking skips them.
_EVENKEEL_NOTE = {"evenkeel_note": True}


def _stop_note(spent: float, budget: float) -> AIMessage:
    return AIMessage(
        f"Stopped by evenkeel: spent ${spent:.4f} of a ${budget:.4f} budget and the increase was declined.",
        response_metadata=dict(_EVENKEEL_NOTE),
    )


def _fixed_bytes(request: ModelRequest) -> int:
    """System prompt and tool schemas: part of every call's input, but not of the message list."""
    size = len(str(request.system_message.content).encode()) if request.system_message else 0
    for tool in request.tools or []:
        try:
            size += len(json.dumps(convert_to_openai_tool(tool), default=str).encode())
        except Exception:  # an unusual tool definition only makes the estimate lower
            continue
    return size


class EvenKeelMiddleware(AgentMiddleware):
    """Classifies each failed model or tool call and acts on it: back off, retry with a hint, fail, or
    escalate to a person. Optionally tracks spend in dollars and pauses the run when over budget."""

    state_schema = EvenKeelState

    def __init__(
        self,
        *,
        classifier: Classifier | None = None,
        policy: FailurePolicy | None = None,
        max_attempts: int = 4,
        budget_usd: float | None = None,
        prices: PriceTable | None = None,
        check_before_call: bool = False,
        assumed_max_output_tokens: int | None = None,
        task: str = "agent",
        sleep: Callable[[float], None] = time.sleep,
        asleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        super().__init__()
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        if budget_usd is not None and prices is None:
            raise ValueError("budget_usd needs prices=PriceTable({...}) to turn tokens into dollars")
        if budget_usd is not None and budget_usd <= 0:
            raise ValueError("budget_usd must be above 0")
        if check_before_call and budget_usd is None:
            raise ValueError("check_before_call needs budget_usd: there is no budget to check against")
        if assumed_max_output_tokens is not None:
            if not check_before_call:
                raise ValueError("assumed_max_output_tokens needs check_before_call=True")
            if assumed_max_output_tokens <= 0:
                raise ValueError("assumed_max_output_tokens must be above 0")
        self._classifier = classifier or RuleClassifier()
        self._policy = policy or default_policy()
        self._max_attempts = max_attempts
        self._budget_usd = budget_usd
        self._prices = prices
        self._check_before_call = check_before_call
        self._assumed_max_output_tokens = assumed_max_output_tokens
        self._warned: set[str] = set()
        self._task = task
        self._sleep = sleep
        self._asleep = asleep

    # Shared helpers

    def _classify(self, ctx: FailureContext) -> FailureVerdict:
        try:
            return self._classifier.classify(ctx)
        except GraphBubbleUp:
            raise
        except Exception:  # a broken classifier must never hide the user's own error
            return replace(RuleClassifier().classify(ctx), source="rules (classifier error)")

    async def _aclassify(self, ctx: FailureContext) -> FailureVerdict:
        try:
            return await self._classifier.aclassify(ctx)
        except GraphBubbleUp:
            raise
        except Exception:
            return replace(RuleClassifier().classify(ctx), source="rules (classifier error)")

    def _context(
        self, error: BaseException, *, step: str, attempt: int, payload: Any = None
    ) -> FailureContext:
        return FailureContext(
            error=error,
            task=self._task,
            attempt=attempt,
            max_attempts=self._max_attempts,
            step=step,
            payload=payload,
            output=getattr(error, "output", None),
        )

    def _model_next(
        self,
        original: ModelRequest,
        current: ModelRequest,
        ctx: FailureContext,
        verdict: FailureVerdict,
        decision: Decision,
        records: list[AttemptRecord],
        escalated: bool,
    ) -> _RetryRequest:
        """Turns a decision into the next request, or raises the original error. A call that still needs a
        person after one approved retry fails, so LangGraph's replay on resume cannot loop forever."""
        action = decision.action
        if isinstance(action, Retry):
            return _RetryRequest(current, action.delay_s, escalated=escalated)
        if isinstance(action, RetryWithHint):
            return _RetryRequest(_with_hint(original, action.hint), escalated=escalated)
        if isinstance(action, Escalate) and not escalated:
            resume = _ask(
                escalation_payload(ctx, verdict, action.reason),
                lambda answer: parse_resume(answer, call_type="model"),
            )
            if resume.decision == "retry":
                request = _with_hint(original, resume.hint) if resume.hint else original
                return _RetryRequest(request, escalated=True)
        attach_attempts(ctx.error, records)
        raise ctx.error

    def _rejected_reply_usd(self, error: BaseException) -> float:
        """What a reply that was billed and then rejected cost: the AIMessage on a LangChain structured
        output error (`ai_message`) or on an evenkeel error (`output`). 0 when the error carries no reply.
        Never raises, so the call's own error is what the policy sees."""
        if self._prices is None:
            return 0.0
        message = getattr(error, "ai_message", None)
        if not isinstance(message, AIMessage):
            message = getattr(error, "output", None)
        if not isinstance(message, AIMessage):
            return 0.0
        try:
            return self._prices.cost_usd(message)
        except Exception as cost_error:
            warnings.warn(
                f"evenkeel could not price a rejected reply, so its cost is not counted: {cost_error}",
                EvenKeelWarning,
                stacklevel=2,
            )
            return 0.0

    @staticmethod
    def _spent_with(request: ModelRequest, rejected_usd: float) -> float | None:
        """The run's spend including rejected replies, or None when there were none to add."""
        if not rejected_usd:
            return None
        return (request.state or {}).get("evenkeel_spent_usd", 0.0) + rejected_usd

    @staticmethod
    def _model_result(
        response: ModelResponse,
        records: list[AttemptRecord],
        new_budget: float | None = None,
        spent_usd: float | None = None,
    ) -> Any:
        update: dict[str, Any] = {}
        if records:
            update["evenkeel_attempts"] = records
        if new_budget is not None:
            update["evenkeel_budget_usd"] = new_budget
        if spent_usd is not None:
            # after_model adds the accepted reply's cost on top of this.
            update["evenkeel_spent_usd"] = spent_usd
        if not update:
            return response
        return ExtendedModelResponse(model_response=response, command=Command(update=update))

    def _next_call_max_usd(self, request: ModelRequest) -> tuple[float | None, str | None]:
        """The estimated most the call can cost, or None and the reason it cannot be estimated. Uses only
        the request and state, so a replay reaches the same result."""
        if self._prices is None:
            return None, "no price table"
        cap = max_output_tokens(request.model, request.model_settings or {})
        cap = cap or self._assumed_max_output_tokens
        if cap is None:
            return None, "no output cap: set max_tokens on the model or assumed_max_output_tokens on evenkeel"
        name = chat_model_name(request.model)
        if name is None:
            return None, "the model does not say which model it calls"
        try:
            price = self._prices.price_for(name)
        except UnknownModelPriceError:
            return None, f"no price for {name!r} in the price table"
        tokens_in = estimate_input_tokens(request.messages, fixed_bytes=_fixed_bytes(request))
        return (tokens_in * price.input_per_mtok + cap * price.output_per_mtok) / 1e6, None

    def _warn_unchecked(self, reason: str) -> None:
        """Say once per reason that check_before_call is on but cannot check, instead of running silently."""
        if reason in self._warned:
            return
        self._warned.add(reason)
        warnings.warn(
            f"evenkeel check_before_call cannot check this model's calls ({reason}). They run unchecked, "
            "and the budget is enforced only after each call, as without check_before_call.",
            EvenKeelWarning,
            stacklevel=2,
        )

    def _check_budget_before_call(self, request: ModelRequest) -> tuple[bool, float | None]:
        """With check_before_call: pause when the call's estimated maximum would cross the budget.
        Returns (go ahead, the budget a reviewer approved, if any)."""
        if not self._check_before_call:
            return True, None
        state = request.state or {}
        budget = state.get("evenkeel_budget_usd", self._budget_usd)
        next_max, reason = self._next_call_max_usd(request)
        if reason is not None:
            self._warn_unchecked(reason)
        if budget is None or next_max is None:
            return True, None
        spent = state.get("evenkeel_spent_usd", 0.0)
        if spent + next_max <= budget:
            return True, None
        new_budget = _ask(
            {
                "type": "evenkeel_budget",
                "spent_usd": round(spent, 6),
                "budget_usd": budget,
                "next_call_max_usd": round(next_max, 6),
                "accepted": [
                    {"approve": True, "budget_usd": "<more than spent_usd + next_call_max_usd>"},
                    {"approve": False},
                ],
            },
            lambda answer: parse_budget_resume(answer, spent_usd=spent, next_call_usd=next_max),
        )
        return new_budget is not None, new_budget

    def _declined(self, request: ModelRequest) -> ModelResponse:
        state = request.state or {}
        budget = state.get("evenkeel_budget_usd", self._budget_usd)
        return ModelResponse(result=[_stop_note(state.get("evenkeel_spent_usd", 0.0), budget)])

    # Model calls

    def wrap_model_call(self, request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]) -> Any:
        go_ahead, new_budget = self._check_budget_before_call(request)
        if not go_ahead:
            return self._declined(request)
        records: list[AttemptRecord] = []
        rejected_usd = 0.0
        current = request
        attempt = 0
        escalated = False
        while True:
            attempt += 1
            try:
                response = handler(current)
            except GraphBubbleUp:
                raise
            except Exception as error:
                rejected_usd += self._rejected_reply_usd(error)
                ctx = self._context(error, step="model", attempt=attempt)
                verdict = self._classify(ctx)
                decision = decide(ctx, verdict, self._policy)
                records.append(decision.record)
                nxt = self._model_next(request, current, ctx, verdict, decision, records, escalated)
                escalated = nxt.escalated
                if nxt.delay_s:
                    self._sleep(nxt.delay_s)
                current = nxt.request
                continue
            return self._model_result(response, records, new_budget, self._spent_with(request, rejected_usd))

    async def awrap_model_call(
        self, request: ModelRequest, handler: Callable[[ModelRequest], Awaitable[ModelResponse]]
    ) -> Any:
        go_ahead, new_budget = self._check_budget_before_call(request)
        if not go_ahead:
            return self._declined(request)
        records: list[AttemptRecord] = []
        rejected_usd = 0.0
        current = request
        attempt = 0
        escalated = False
        while True:
            attempt += 1
            try:
                response = await handler(current)
            except GraphBubbleUp:
                raise
            except Exception as error:
                rejected_usd += self._rejected_reply_usd(error)
                ctx = self._context(error, step="model", attempt=attempt)
                verdict = await self._aclassify(ctx)
                decision = decide(ctx, verdict, self._policy)
                records.append(decision.record)
                nxt = self._model_next(request, current, ctx, verdict, decision, records, escalated)
                escalated = nxt.escalated
                if nxt.delay_s:
                    await self._asleep(nxt.delay_s)
                current = nxt.request
                continue
            return self._model_result(response, records, new_budget, self._spent_with(request, rejected_usd))

    # Tool calls

    def _tool_next(
        self,
        request: ToolCallRequest,
        ctx: FailureContext,
        verdict: FailureVerdict,
        decision: Decision,
        records: list[AttemptRecord],
        escalated: bool,
    ) -> _RetryRequest | Command:
        """Retry the tool, hand the error back to the model, or raise the original error."""
        call = request.tool_call
        action = decision.action

        def back_to_model(content: str, status: str = "error") -> Command:
            records[-1] = {**records[-1], "action": "returned_to_model"}
            message = ToolMessage(content=content, tool_call_id=call["id"], name=call["name"], status=status)
            return Command(update={"messages": [message], "evenkeel_attempts": records})

        if isinstance(action, Retry):
            return _RetryRequest(request, action.delay_s, escalated=escalated)
        if isinstance(action, RetryWithHint):
            return back_to_model(f"Error: {error_message(ctx.error)}. {action.hint}")
        if verdict.kind == "bad_input":
            return back_to_model(
                f"Error: {error_message(ctx.error)}. Fix the arguments and call the tool again."
            )
        if isinstance(action, Escalate) and not escalated:
            resume = _ask(
                escalation_payload(ctx, verdict, action.reason),
                lambda answer: parse_resume(answer, call_type="tool"),
            )
            if resume.decision == "retry" and resume.hint:
                # A tool cannot take a hint, so the reviewer's words go to the model, which can call again.
                return back_to_model(f"Error: {error_message(ctx.error)}. A reviewer said: {resume.hint}")
            if resume.decision == "retry":
                return _RetryRequest(request, escalated=True)
            if resume.decision == "skip":
                records[-1] = {**records[-1], "action": "skipped_by_reviewer"}
                message = ToolMessage(content=resume.message, tool_call_id=call["id"], name=call["name"])
                return Command(update={"messages": [message], "evenkeel_attempts": records})
        attach_attempts(ctx.error, records)
        raise ctx.error

    @staticmethod
    def _tool_result(result: Any, records: list[AttemptRecord]) -> Any:
        if not records:
            return result
        if isinstance(result, ToolMessage):
            return Command(update={"messages": [result], "evenkeel_attempts": records})
        if isinstance(result, Command) and isinstance(result.update, dict):
            existing = list(result.update.get("evenkeel_attempts", []))
            return Command(
                graph=result.graph,
                update={**result.update, "evenkeel_attempts": existing + records},
                resume=result.resume,
                goto=result.goto,
            )
        return result

    def wrap_tool_call(self, request: ToolCallRequest, handler: Callable[[ToolCallRequest], Any]) -> Any:
        records: list[AttemptRecord] = []
        attempt = 0
        escalated = False
        while True:
            attempt += 1
            try:
                result = handler(request)
            except GraphBubbleUp:
                raise
            except Exception as error:
                ctx = self._context(
                    error,
                    step=f"tool:{request.tool_call['name']}",
                    attempt=attempt,
                    payload=request.tool_call.get("args"),
                )
                verdict = self._classify(ctx)
                decision = decide(ctx, verdict, self._policy)
                records.append(decision.record)
                nxt = self._tool_next(request, ctx, verdict, decision, records, escalated)
                if isinstance(nxt, Command):
                    return nxt
                escalated = nxt.escalated
                if nxt.delay_s:
                    self._sleep(nxt.delay_s)
                continue
            return self._tool_result(result, records)

    async def awrap_tool_call(
        self, request: ToolCallRequest, handler: Callable[[ToolCallRequest], Awaitable[Any]]
    ) -> Any:
        records: list[AttemptRecord] = []
        attempt = 0
        escalated = False
        while True:
            attempt += 1
            try:
                result = await handler(request)
            except GraphBubbleUp:
                raise
            except Exception as error:
                ctx = self._context(
                    error,
                    step=f"tool:{request.tool_call['name']}",
                    attempt=attempt,
                    payload=request.tool_call.get("args"),
                )
                verdict = await self._aclassify(ctx)
                decision = decide(ctx, verdict, self._policy)
                records.append(decision.record)
                nxt = self._tool_next(request, ctx, verdict, decision, records, escalated)
                if isinstance(nxt, Command):
                    return nxt
                escalated = nxt.escalated
                if nxt.delay_s:
                    await self._asleep(nxt.delay_s)
                continue
            return self._tool_result(result, records)

    # Budget

    def before_agent(self, state: EvenKeelState, runtime: Any) -> dict[str, Any] | None:
        """Each run starts with zero spend and the configured budget; state persists per thread."""
        if self._prices is None:
            return None
        reset: dict[str, Any] = {"evenkeel_spent_usd": 0.0}
        if self._budget_usd is not None:
            reset["evenkeel_budget_usd"] = self._budget_usd
        return reset

    def after_model(self, state: EvenKeelState, runtime: Any) -> dict[str, Any] | None:
        if self._prices is None:
            return None
        # The model's reply is the newest AIMessage; structured output can put a ToolMessage after it.
        message = next((m for m in reversed(state["messages"]) if isinstance(m, AIMessage)), None)
        if message is None or (message.response_metadata or {}).get("evenkeel_note"):
            return None
        spent = state.get("evenkeel_spent_usd", 0.0) + self._prices.cost_usd(message)
        return {"evenkeel_spent_usd": spent}

    @hook_config(can_jump_to=["end"])
    def before_model(self, state: EvenKeelState, runtime: Any) -> dict[str, Any] | None:
        budget = state.get("evenkeel_budget_usd", self._budget_usd)
        if budget is None:
            return None
        spent = state.get("evenkeel_spent_usd", 0.0)
        if spent < budget:
            return None
        new_budget = _ask(
            {
                "type": "evenkeel_budget",
                "spent_usd": round(spent, 6),
                "budget_usd": budget,
                "accepted": [{"approve": True, "budget_usd": "<more than spent_usd>"}, {"approve": False}],
            },
            lambda answer: parse_budget_resume(answer, spent_usd=spent),
        )
        if new_budget is not None:
            return {"evenkeel_budget_usd": new_budget}
        return {"messages": [_stop_note(spent, budget)], "jump_to": "end"}
