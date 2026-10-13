import pytest

from langchain_evenkeel.failure import (
    BadInputError,
    BadOutputError,
    Escalate,
    Fail,
    FailureContext,
    FailureVerdict,
    NeedsHumanError,
    OverBudgetError,
    Retry,
    RetryWithHint,
    RuleClassifier,
    default_policy,
)


def with_attrs(cls: type[Exception], message: str, **attrs) -> Exception:
    error = cls(message)
    for key, value in attrs.items():
        setattr(error, key, value)
    return error


def named(name: str, message: str = "x") -> Exception:
    return type(name, (Exception,), {})(message)


class FakeResponse:
    def __init__(self, status_code: int) -> None:
        self.status_code = status_code


def caused_by(outer: Exception, cause: BaseException) -> Exception:
    outer.__cause__ = cause
    return outer


CASES = [
    ("503 from an upstream API", with_attrs(Exception, "HTTP 503", status_code=503), "transient"),
    ("429 rate limit", with_attrs(Exception, "rate limited", status_code=429), "transient"),
    ("status instead of status_code", with_attrs(Exception, "x", status=502), "transient"),
    ("status on the response (httpx)", with_attrs(Exception, "x", response=FakeResponse(504)), "transient"),
    ("socket timeout", TimeoutError("timed out"), "transient"),
    ("connection reset", ConnectionResetError("reset"), "transient"),
    ("SDK timeout by name", named("APITimeoutError"), "transient"),
    ("SDK connection error by name", named("APIConnectionError"), "transient"),
    (
        "network error on the cause",
        caused_by(RuntimeError("request failed"), ConnectionRefusedError()),
        "transient",
    ),
    ("422 unprocessable input", with_attrs(Exception, "x", status_code=422), "bad_input"),
    ("pydantic validation failure", named("ValidationError"), "bad_input"),
    ("model returned unusable output", BadOutputError("tool name does not exist"), "bad_output"),
    ("handler asks for a human", NeedsHumanError("refund over limit"), "needs_human"),
    ("explicit bad input", BadInputError("missing field"), "bad_input"),
    ("explicit over budget", OverBudgetError("spent too much"), "over_budget"),
    ("401 bad credentials", with_attrs(Exception, "x", status_code=401), "fatal"),
    ("plain bug", TypeError("'NoneType' object is not subscriptable"), "fatal"),
    ("unknown cause", caused_by(RuntimeError("request failed"), KeyError("k")), "fatal"),
]


def ctx(error: BaseException, attempt: int = 1) -> FailureContext:
    return FailureContext(error=error, task="t", attempt=attempt, max_attempts=3)


@pytest.mark.parametrize(("label", "error", "kind"), CASES, ids=[c[0] for c in CASES])
def test_rule_classifier(label, error, kind):
    verdict = RuleClassifier().classify(ctx(error))
    assert verdict.kind == kind
    assert 0 < verdict.confidence <= 1
    assert verdict.source == "rules"


def test_explicit_error_classes_are_certain():
    assert RuleClassifier().classify(ctx(BadOutputError("x"))).confidence == 1.0


def test_aclassify_matches_classify():
    import asyncio

    error = with_attrs(Exception, "x", status_code=503)
    assert asyncio.run(RuleClassifier().aclassify(ctx(error))) == RuleClassifier().classify(ctx(error))


def test_errors_carry_output():
    assert BadOutputError("x", output={"tool": "serch_web"}).output == {"tool": "serch_web"}


def test_policy_transient_backs_off_with_full_jitter():
    policy = default_policy(base_s=1.0, max_s=60.0, random=lambda: 0.5)
    first = policy(FailureVerdict("transient", 0.9), ctx(TimeoutError(), attempt=1))
    third = policy(FailureVerdict("transient", 0.9), ctx(TimeoutError(), attempt=3))
    assert first == Retry(delay_s=0.5)
    assert third == Retry(delay_s=2.0)


def test_policy_backoff_is_capped():
    policy = default_policy(base_s=1.0, max_s=5.0, random=lambda: 1.0)
    assert policy(FailureVerdict("transient", 0.9), ctx(TimeoutError(), attempt=10)) == Retry(delay_s=5.0)


def test_policy_bad_output_retries_with_the_error_as_a_hint():
    action = default_policy()(
        FailureVerdict("bad_output", 1.0), ctx(BadOutputError("unknown tool serch_web"))
    )
    assert action == RetryWithHint(
        hint="The previous attempt produced unusable output: unknown tool serch_web"
    )
    assert action.type == "retry_modified"


@pytest.mark.parametrize("kind", ["needs_human", "over_budget"])
def test_policy_escalates(kind):
    action = default_policy()(FailureVerdict(kind, 1.0), ctx(NeedsHumanError("refund over limit")))
    assert action == Escalate(reason="refund over limit")


@pytest.mark.parametrize("kind", ["bad_input", "fatal"])
def test_policy_fails(kind):
    action = default_policy()(FailureVerdict(kind, 0.8), ctx(TypeError("bug")))
    assert action == Fail(reason="bug")


def test_policy_names_errors_without_a_message():
    action = default_policy()(FailureVerdict("fatal", 0.5), ctx(KeyError()))
    assert action == Fail(reason="KeyError")


def test_temporary_dns_failure_is_transient():
    import socket

    error = socket.gaierror(socket.EAI_AGAIN, "Temporary failure in name resolution")
    assert RuleClassifier().classify(ctx(error)).kind == "transient"


@pytest.mark.parametrize("code", ["ETIMEDOUT", "ECONNRESET", "ECONNREFUSED", "EPIPE", "EAI_AGAIN"])
def test_network_error_codes_are_transient_as_in_keel(code):
    error = with_attrs(Exception, f"connect {code}", code=code)
    assert RuleClassifier().classify(ctx(error)).kind == "transient"


def test_network_error_code_on_the_cause_is_transient():
    error = caused_by(RuntimeError("request failed"), with_attrs(Exception, "x", code="ECONNRESET"))
    assert RuleClassifier().classify(ctx(error)).kind == "transient"
