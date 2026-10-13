"""JevClassifier: asks TypeSafe's Jev what kind of failure an error is, with the rule classifier as fallback.
Ported from keel's src/jev-classifier.ts. Needs the `jev` extra (typesafe-sdk): pip install langchain-evenkeel[jev]."""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import replace
from typing import Any

from .failure import FailureContext, FailureVerdict, RuleClassifier, error_message

logger = logging.getLogger(__name__)

_PREVIEW_CHARS = 2000

# over_budget is never asked: the middleware raises it explicitly.
_INSTRUCTIONS = (
    "A step of an automated job, often an AI agent, failed with `error`. Decide what kind of failure it is, "
    "so the job engine can retry, retry with a correction, stop, or ask a person. Judge from `error`, `task` "
    "and `payload`, and from `step` (the step that was running) and `output` (what the model or tool produced) "
    "when present."
)
_CRITERIA = {
    "transient": (
        "A temporary problem outside the job: rate limits, timeouts, overloaded or unavailable services, "
        "dropped connections. Retrying the same request later is likely to succeed."
    ),
    "bad_input": (
        "The job's own input is invalid or cannot be processed: missing or malformed fields, values that fail "
        "validation, content too long for the model's context window. Retrying with the same input will fail again."
    ),
    "bad_output": (
        "A model or tool produced unusable output: invalid JSON, a call to a tool or function that does not exist, "
        "wrong or missing arguments, a response that breaks the required schema. Retrying with a correction to "
        "the model is likely to help."
    ),
    "needs_human": (
        "The job cannot continue until a person decides: an action above an approval limit, a model refusal or "
        "policy block that needs review, information or permission that only a person can provide."
    ),
    "fatal": (
        "A bug or permanent problem in the code or configuration: type errors, undefined values, invalid "
        "credentials, missing configuration or resources. Retrying will not help."
    ),
}
_ASKED = frozenset(_CRITERIA)


def _question() -> Any:
    from typesafe_sdk import Choice

    return Choice(instructions=_INSTRUCTIONS, criteria=dict(_CRITERIA))


def _read_answer(response: Any) -> tuple[str | None, float]:
    answers = getattr(response, "answers", None) or {}
    answer = answers.get("failure_kind")
    choice = getattr(answer, "choice", None)
    confidence = getattr(answer, "confidence", None)
    return (
        choice if isinstance(choice, str) else None,
        float(confidence) if isinstance(confidence, (int, float)) else 0.0,
    )


def _json(value: Any) -> str:
    try:
        return json.dumps(value, default=str)
    except (TypeError, ValueError):
        return str(value)


def _preview(text: str) -> str:
    return f"{text[:_PREVIEW_CHARS]}... (truncated)" if len(text) > _PREVIEW_CHARS else text


def _describe_cause(cause: BaseException) -> str:
    return f"{type(cause).__name__}: {error_message(cause)}"


def describe_failure(ctx: FailureContext) -> dict[str, Any]:
    """The state sent to Jev: the error with its HTTP status, code and cause, plus the run context."""
    error = ctx.error
    described: dict[str, Any] = {"name": type(error).__name__, "message": error_message(error)}
    for attr in ("status_code", "status", "code"):
        value = getattr(error, attr, None)
        if value is not None:
            described["status" if attr != "code" else "code"] = value
    cause = error.__cause__ or error.__context__
    if cause is not None:
        described["cause"] = _describe_cause(cause)
    state: dict[str, Any] = {
        "task": ctx.task,
        "attempt": f"{ctx.attempt} of {ctx.max_attempts}",
        "error": described,
        "payload": _preview(_json(ctx.payload)),
    }
    if ctx.step is not None:
        state["step"] = ctx.step
    if ctx.output is not None:
        state["output"] = _preview(_json(ctx.output))
    return state


class JevClassifier:
    """Cascade: explicit signals (rule confidence 1) are classified by rules for free. Everything else is
    asked of Jev, falling back to rules when Jev is unsure, unusable or unavailable."""

    def __init__(
        self,
        client: Any = None,
        *,
        async_client: Any = None,
        min_confidence: float = 0.5,
        model: str | None = None,
    ) -> None:
        if client is None and async_client is None:
            raise ValueError(
                "JevClassifier needs a client: JevClassifier(TypeSafeClient()) or async_client=..."
            )
        self._rules = RuleClassifier()
        self._client = client
        self._async_client = async_client
        self._min_confidence = min_confidence
        self._extra = {"model": model} if model else {}

    def classify(self, ctx: FailureContext) -> FailureVerdict:
        by_rules = self._rules.classify(ctx)
        if by_rules.confidence >= 1:
            return by_rules
        if self._client is None:
            return asyncio.run(self.aclassify(ctx))
        try:
            response = self._client.system_one(
                state=describe_failure(ctx), questions={"failure_kind": _question()}, **self._extra
            )
        except Exception as error:  # an outage must not turn every failure into a classifier error
            logger.warning("Jev classification failed, using rules: %s", error)
            return replace(by_rules, source="rules (jev unavailable)")
        return self._verdict(response, by_rules)

    async def aclassify(self, ctx: FailureContext) -> FailureVerdict:
        by_rules = self._rules.classify(ctx)
        if by_rules.confidence >= 1:
            return by_rules
        if self._async_client is None:
            return await asyncio.to_thread(self.classify, ctx)
        try:
            response = await self._async_client.system_one(
                state=describe_failure(ctx), questions={"failure_kind": _question()}, **self._extra
            )
        except Exception as error:
            logger.warning("Jev classification failed, using rules: %s", error)
            return replace(by_rules, source="rules (jev unavailable)")
        return self._verdict(response, by_rules)

    def _verdict(self, response: Any, by_rules: FailureVerdict) -> FailureVerdict:
        choice, confidence = _read_answer(response)
        if choice not in _ASKED:
            return replace(by_rules, source="rules (jev answer unusable)")
        if confidence < self._min_confidence:
            return replace(by_rules, source="rules (jev unsure)")
        return FailureVerdict(choice, confidence, "jev")  # type: ignore[arg-type]
