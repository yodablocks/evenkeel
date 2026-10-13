"""Failure kinds, the rule classifier and the default policy. Ported from keel's src/failure.ts."""

from __future__ import annotations

import random as _random
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

FailureKind = Literal["transient", "bad_input", "bad_output", "needs_human", "over_budget", "fatal"]


@dataclass(frozen=True)
class FailureContext:
    error: BaseException
    task: str
    attempt: int
    """The attempt that just failed, starting at 1."""
    max_attempts: int
    step: str | None = None
    """What was running: "model" or "tool:<name>"."""
    payload: Any = None
    output: Any = None


@dataclass(frozen=True)
class FailureVerdict:
    kind: FailureKind
    confidence: float
    """0 to 1. Rule matches on explicit signals are near 1, fallbacks are low."""
    source: str = "rules"
    """Which classifier decided, e.g. "rules", "jev", "rules (jev unavailable)"."""


class Classifier(Protocol):
    def classify(self, ctx: FailureContext) -> FailureVerdict: ...

    async def aclassify(self, ctx: FailureContext) -> FailureVerdict: ...


class EvenKeelError(Exception):
    """Base for explicit failure signals. `output` is what the model or tool produced."""

    def __init__(self, message: str, *, output: Any = None) -> None:
        super().__init__(message)
        self.output = output


class BadInputError(EvenKeelError):
    """The input can never succeed, whatever the retry."""


class BadOutputError(EvenKeelError):
    """A model or tool produced unusable output (hallucinated tool, malformed JSON)."""


class NeedsHumanError(EvenKeelError):
    """A person has to decide before the run can continue."""


class OverBudgetError(EvenKeelError):
    """The run has spent its budget."""


_TRANSIENT_NAMES = frozenset(
    {
        "APITimeoutError",
        "APIConnectionError",
        "ConnectTimeout",
        "ReadTimeout",
        "WriteTimeout",
        "PoolTimeout",
        "TimeoutException",
        "ConnectError",
        "ReadError",
        "RemoteProtocolError",
    }
)
_VALIDATION_NAMES = frozenset({"ValidationError"})


def error_message(error: BaseException) -> str:
    return str(error) or type(error).__name__


def _http_status(error: BaseException) -> int | None:
    for attr in ("status_code", "status"):
        value = getattr(error, attr, None)
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    value = getattr(getattr(error, "response", None), "status_code", None)
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _is_network_error(error: BaseException) -> bool:
    return isinstance(error, (TimeoutError, ConnectionError)) or type(error).__name__ in _TRANSIENT_NAMES


class RuleClassifier:
    """Classifies failures from explicit signals only: error classes, HTTP status, network errors."""

    def classify(self, ctx: FailureContext) -> FailureVerdict:
        error = ctx.error
        if isinstance(error, BadInputError):
            return FailureVerdict("bad_input", 1.0)
        if isinstance(error, BadOutputError):
            return FailureVerdict("bad_output", 1.0)
        if isinstance(error, NeedsHumanError):
            return FailureVerdict("needs_human", 1.0)
        if isinstance(error, OverBudgetError):
            return FailureVerdict("over_budget", 1.0)

        status = _http_status(error)
        if status is not None:
            if status in (408, 429) or status >= 500:
                return FailureVerdict("transient", 0.9)
            if status in (400, 422):
                return FailureVerdict("bad_input", 0.8)
            return FailureVerdict("fatal", 0.8)
        if _is_network_error(error):
            return FailureVerdict("transient", 0.9)
        # SDKs often wrap the network error: look one level down the chain.
        cause = error.__cause__ or error.__context__
        if cause is not None and _is_network_error(cause):
            return FailureVerdict("transient", 0.9)
        if type(error).__name__ in _VALIDATION_NAMES:
            return FailureVerdict("bad_input", 0.7)
        return FailureVerdict("fatal", 0.5)

    async def aclassify(self, ctx: FailureContext) -> FailureVerdict:
        return self.classify(ctx)


@dataclass(frozen=True)
class Retry:
    delay_s: float
    type: Literal["retry"] = field(default="retry", init=False)


@dataclass(frozen=True)
class RetryWithHint:
    hint: str
    type: Literal["retry_modified"] = field(default="retry_modified", init=False)


@dataclass(frozen=True)
class Escalate:
    reason: str
    type: Literal["escalate"] = field(default="escalate", init=False)


@dataclass(frozen=True)
class Fail:
    reason: str
    type: Literal["fail"] = field(default="fail", init=False)


FailureAction = Retry | RetryWithHint | Escalate | Fail
FailurePolicy = Callable[[FailureVerdict, FailureContext], FailureAction]


def default_policy(
    *, base_s: float = 1.0, max_s: float = 60.0, random: Callable[[], float] = _random.random
) -> FailurePolicy:
    """Transient: exponential backoff with full jitter. Bad output: retry with the error as a hint.
    Needs-human and over-budget: escalate. Bad input and fatal: fail."""

    def policy(verdict: FailureVerdict, ctx: FailureContext) -> FailureAction:
        message = error_message(ctx.error)
        if verdict.kind == "transient":
            ceiling = min(max_s, base_s * 2 ** (ctx.attempt - 1))
            return Retry(delay_s=random() * ceiling)
        if verdict.kind == "bad_output":
            return RetryWithHint(hint=f"The previous attempt produced unusable output: {message}")
        if verdict.kind in ("needs_human", "over_budget"):
            return Escalate(reason=message)
        return Fail(reason=message)

    return policy
