"""EvenKeelMiddleware: classified retries with hints, dollar budgets and escalation for LangChain agents."""

from __future__ import annotations

import asyncio
import operator
import time
from collections.abc import Awaitable, Callable
from dataclasses import replace
from typing import Annotated, Any

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from typing_extensions import NotRequired

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
from .budget import PriceTable
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
        self._classifier = classifier or RuleClassifier()
        self._policy = policy or default_policy()
        self._max_attempts = max_attempts
        self._budget_usd = budget_usd
        self._prices = prices
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

    @staticmethod
    def _model_result(response: ModelResponse, records: list[AttemptRecord]) -> Any:
        if not records:
            return response
        return ExtendedModelResponse(
            model_response=response, command=Command(update={"evenkeel_attempts": records})
        )

    # Model calls

    def wrap_model_call(self, request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]) -> Any:
        records: list[AttemptRecord] = []
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
            return self._model_result(response, records)

    async def awrap_model_call(
        self, request: ModelRequest, handler: Callable[[ModelRequest], Awaitable[ModelResponse]]
    ) -> Any:
        records: list[AttemptRecord] = []
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
            return self._model_result(response, records)

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
        if message is None:
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
        note = AIMessage(
            f"Stopped by evenkeel: spent ${spent:.4f} of a ${budget:.4f} budget "
            "and the increase was declined."
        )
        return {"messages": [note], "jump_to": "end"}
