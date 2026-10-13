"""EvenKeelMiddleware: classified retries with hints, dollar budgets and escalation for LangChain agents."""

from __future__ import annotations

import asyncio
import operator
import time
from collections.abc import Awaitable, Callable
from typing import Annotated, Any

from langchain_core.messages import HumanMessage, ToolMessage
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
    interrupt,
)
from ._core import AttemptRecord, Decision, attach_attempts, decide, escalation_payload, parse_resume
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

    def __init__(self, request: Any, delay_s: float = 0.0) -> None:
        self.request = request
        self.delay_s = delay_s


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
    ) -> _RetryRequest:
        """Turns a decision into the next request, or raises the original error."""
        action = decision.action
        if isinstance(action, Retry):
            return _RetryRequest(current, action.delay_s)
        if isinstance(action, RetryWithHint):
            return _RetryRequest(_with_hint(original, action.hint))
        if isinstance(action, Escalate):
            resume = parse_resume(
                interrupt(escalation_payload(ctx, verdict, action.reason)), call_type="model"
            )
            if resume.decision == "retry":
                return _RetryRequest(_with_hint(original, resume.hint) if resume.hint else original)
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
        while True:
            attempt += 1
            try:
                response = handler(current)
            except GraphBubbleUp:
                raise
            except Exception as error:
                ctx = self._context(error, step="model", attempt=attempt)
                verdict = self._classifier.classify(ctx)
                decision = decide(ctx, verdict, self._policy)
                records.append(decision.record)
                nxt = self._model_next(request, current, ctx, verdict, decision, records)
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
        while True:
            attempt += 1
            try:
                response = await handler(current)
            except GraphBubbleUp:
                raise
            except Exception as error:
                ctx = self._context(error, step="model", attempt=attempt)
                verdict = await self._classifier.aclassify(ctx)
                decision = decide(ctx, verdict, self._policy)
                records.append(decision.record)
                nxt = self._model_next(request, current, ctx, verdict, decision, records)
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
    ) -> _RetryRequest | Command:
        """Retry the tool, hand the error back to the model, or raise the original error."""
        call = request.tool_call
        action = decision.action

        def back_to_model(content: str, status: str = "error") -> Command:
            records[-1] = {**records[-1], "action": "returned_to_model"}
            message = ToolMessage(content=content, tool_call_id=call["id"], name=call["name"], status=status)
            return Command(update={"messages": [message], "evenkeel_attempts": records})

        if isinstance(action, Retry):
            return _RetryRequest(request, action.delay_s)
        if isinstance(action, RetryWithHint):
            return back_to_model(f"Error: {error_message(ctx.error)}. {action.hint}")
        if verdict.kind == "bad_input":
            return back_to_model(
                f"Error: {error_message(ctx.error)}. Fix the arguments and call the tool again."
            )
        if isinstance(action, Escalate):
            resume = parse_resume(
                interrupt(escalation_payload(ctx, verdict, action.reason)), call_type="tool"
            )
            if resume.decision == "retry":
                return _RetryRequest(request)
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
                verdict = self._classifier.classify(ctx)
                decision = decide(ctx, verdict, self._policy)
                records.append(decision.record)
                nxt = self._tool_next(request, ctx, verdict, decision, records)
                if isinstance(nxt, Command):
                    return nxt
                if nxt.delay_s:
                    self._sleep(nxt.delay_s)
                continue
            return self._tool_result(result, records)

    async def awrap_tool_call(
        self, request: ToolCallRequest, handler: Callable[[ToolCallRequest], Awaitable[Any]]
    ) -> Any:
        records: list[AttemptRecord] = []
        attempt = 0
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
                verdict = await self._classifier.aclassify(ctx)
                decision = decide(ctx, verdict, self._policy)
                records.append(decision.record)
                nxt = self._tool_next(request, ctx, verdict, decision, records)
                if isinstance(nxt, Command):
                    return nxt
                if nxt.delay_s:
                    await self._asleep(nxt.delay_s)
                continue
            return self._tool_result(result, records)
