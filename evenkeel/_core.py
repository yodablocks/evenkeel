"""Decision logic shared by every integration: apply the policy, enforce the attempt limit, build records,
and read reviewers' resume values. No LangChain imports, so other integrations can reuse it."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from typing_extensions import TypedDict

from .failure import (
    Escalate,
    Fail,
    FailureAction,
    FailureContext,
    FailurePolicy,
    FailureVerdict,
    Retry,
    RetryWithHint,
    error_message,
)

_RECORD_ERROR_CHARS = 500


class AttemptRecord(TypedDict):
    attempt: int
    call: str
    error_type: str
    error: str
    kind: str
    confidence: float
    source: str
    action: str


@dataclass(frozen=True)
class Decision:
    action: FailureAction
    record: AttemptRecord


def decide(ctx: FailureContext, verdict: FailureVerdict, policy: FailurePolicy) -> Decision:
    action = policy(verdict, ctx)
    if isinstance(action, (Retry, RetryWithHint, Escalate)) and ctx.attempt >= ctx.max_attempts:
        action = Fail(reason=f"attempt limit reached ({ctx.max_attempts}): {error_message(ctx.error)}")
    record = AttemptRecord(
        attempt=ctx.attempt,
        call=ctx.step or "",
        error_type=type(ctx.error).__name__,
        error=error_message(ctx.error)[:_RECORD_ERROR_CHARS],
        kind=verdict.kind,
        confidence=round(verdict.confidence, 3),
        source=verdict.source,
        action=action.type,
    )
    return Decision(action=action, record=record)


def escalation_payload(ctx: FailureContext, verdict: FailureVerdict, reason: str) -> dict[str, Any]:
    is_tool = (ctx.step or "").startswith("tool:")
    return {
        "type": "evenkeel_escalation",
        "call": ctx.step or "",
        "error_type": type(ctx.error).__name__,
        "error": error_message(ctx.error)[:_RECORD_ERROR_CHARS],
        "kind": verdict.kind,
        "confidence": round(verdict.confidence, 3),
        "reason": reason,
        "accepted": ["retry", "fail", "skip"] if is_tool else ["retry", "fail"],
    }


@dataclass(frozen=True)
class Resume:
    decision: Literal["retry", "fail", "skip"]
    hint: str | None
    message: str | None


_RESUME_HELP = (
    'Resume with {"decision": "retry"} (optionally with "hint": "..."), {"decision": "fail"}, '
    'or, for tool calls only, {"decision": "skip", "message": "..."}'
)


def parse_resume(value: Any, *, call_type: Literal["model", "tool"]) -> Resume:
    if not isinstance(value, dict) or value.get("decision") not in ("retry", "fail", "skip"):
        raise ValueError(f"Unrecognised escalation decision {value!r}. {_RESUME_HELP}")
    decision = value["decision"]
    hint = value.get("hint")
    message = value.get("message")
    if hint is not None and not isinstance(hint, str):
        raise ValueError(f"The decision's hint must be a string, got {hint!r}. {_RESUME_HELP}")
    if decision == "skip":
        if call_type == "model":
            raise ValueError(f"A model call cannot be skipped: it has no stand-in result. {_RESUME_HELP}")
        if not isinstance(message, str):
            raise ValueError(f"A skip decision needs a message for the model. {_RESUME_HELP}")
        return Resume("skip", None, message)
    return Resume(decision, hint, None)


_BUDGET_HELP = 'Resume with {"approve": true, "budget_usd": <more than spent>} or {"approve": false}'


def parse_budget_resume(value: Any, *, spent_usd: float) -> float | None:
    if not isinstance(value, dict) or not isinstance(value.get("approve"), bool):
        raise ValueError(f"Unrecognised budget decision {value!r}. {_BUDGET_HELP}")
    if not value["approve"]:
        return None
    budget = value.get("budget_usd")
    if not isinstance(budget, (int, float)) or isinstance(budget, bool) or budget <= spent_usd:
        raise ValueError(
            f"An approval needs a budget_usd above the ${spent_usd:.4f} already spent, "
            f"got {budget!r}. {_BUDGET_HELP}"
        )
    return float(budget)


def attach_attempts(error: BaseException, records: list[AttemptRecord]) -> None:
    try:
        error.evenkeel_attempts = list(records)  # type: ignore[attr-defined]
    except AttributeError:  # some built-in exceptions refuse new attributes
        if hasattr(error, "add_note"):  # Python 3.11+
            error.add_note(f"evenkeel attempts: {records}")
