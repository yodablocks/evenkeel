import pytest

from langchain_evenkeel._core import (
    Resume,
    attach_attempts,
    decide,
    escalation_payload,
    parse_budget_resume,
    parse_resume,
)
from langchain_evenkeel.failure import (
    Fail,
    FailureContext,
    FailureVerdict,
    NeedsHumanError,
    Retry,
    RetryWithHint,
    default_policy,
)


def ctx(error: BaseException, attempt: int = 1, max_attempts: int = 3) -> FailureContext:
    return FailureContext(error=error, task="agent", attempt=attempt, max_attempts=max_attempts, step="model")


POLICY = default_policy(random=lambda: 0.5)


def test_decide_records_the_attempt():
    decision = decide(ctx(TimeoutError("timed out")), FailureVerdict("transient", 0.9), POLICY)
    assert isinstance(decision.action, Retry)
    assert decision.record == {
        "attempt": 1,
        "call": "model",
        "error_type": "TimeoutError",
        "error": "timed out",
        "kind": "transient",
        "confidence": 0.9,
        "source": "rules",
        "action": "retry",
    }


@pytest.mark.parametrize("kind", ["transient", "bad_output"])
def test_retries_stop_at_the_attempt_limit(kind):
    decision = decide(ctx(TimeoutError("still down"), attempt=3), FailureVerdict(kind, 0.9), POLICY)
    assert decision.action == Fail(reason="attempt limit reached (3): still down")
    assert decision.record["action"] == "fail"


def test_long_error_messages_are_truncated_in_records():
    decision = decide(ctx(TypeError("x" * 2000)), FailureVerdict("fatal", 0.5), POLICY)
    assert len(decision.record["error"]) == 500


def test_hint_actions_keep_their_hint():
    decision = decide(ctx(TypeError("bad json")), FailureVerdict("bad_output", 1.0), POLICY)
    assert isinstance(decision.action, RetryWithHint) and "bad json" in decision.action.hint


def test_escalation_payload_is_json_ready():
    payload = escalation_payload(
        ctx(NeedsHumanError("refund over limit")), FailureVerdict("needs_human", 1.0), "refund over limit"
    )
    assert payload == {
        "type": "evenkeel_escalation",
        "call": "model",
        "error_type": "NeedsHumanError",
        "error": "refund over limit",
        "kind": "needs_human",
        "confidence": 1.0,
        "reason": "refund over limit",
        "accepted": ["retry", "fail"],
    }


def test_escalation_payload_offers_skip_for_tools():
    tool_ctx = FailureContext(
        error=NeedsHumanError("x"), task="agent", attempt=1, max_attempts=3, step="tool:pay"
    )
    assert escalation_payload(tool_ctx, FailureVerdict("needs_human", 1.0), "x")["accepted"] == [
        "retry",
        "fail",
        "skip",
    ]


def test_parse_resume_accepts_the_documented_values():
    assert parse_resume({"decision": "retry"}, call_type="model") == Resume("retry", None, None)
    assert parse_resume({"decision": "retry", "hint": "use EUR"}, call_type="model") == Resume(
        "retry", "use EUR", None
    )
    assert parse_resume({"decision": "fail"}, call_type="tool") == Resume("fail", None, None)
    assert parse_resume({"decision": "skip", "message": "done by hand"}, call_type="tool") == Resume(
        "skip", None, "done by hand"
    )


@pytest.mark.parametrize(
    ("value", "call_type"),
    [
        ("yes", "model"),
        ({}, "model"),
        ({"decision": "approve"}, "model"),
        ({"decision": "skip", "message": "x"}, "model"),
        ({"decision": "skip"}, "tool"),
        ({"decision": "retry", "hint": 3}, "model"),
    ],
)
def test_parse_resume_rejects_anything_else_with_the_accepted_values(value, call_type):
    with pytest.raises(ValueError, match="decision"):
        parse_resume(value, call_type=call_type)


def test_parse_budget_resume():
    assert parse_budget_resume({"approve": True, "budget_usd": 0.5}, spent_usd=0.4) == 0.5
    assert parse_budget_resume({"approve": False}, spent_usd=0.4) is None


@pytest.mark.parametrize(
    "value",
    [
        "yes",
        {"approve": True},
        {"approve": True, "budget_usd": 0.3},
        {"approve": True, "budget_usd": "1"},
        {},
    ],
)
def test_parse_budget_resume_rejects_bad_values(value):
    with pytest.raises(ValueError, match="approve"):
        parse_budget_resume(value, spent_usd=0.4)


def test_attach_attempts_sets_an_attribute_on_the_error():
    error = TypeError("bug")
    attach_attempts(error, [{"attempt": 1}])
    assert error.evenkeel_attempts == [{"attempt": 1}]
