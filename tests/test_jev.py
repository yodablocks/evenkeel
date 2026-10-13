import asyncio
import json
from types import SimpleNamespace

import pytest

from langchain_evenkeel.failure import BadOutputError, FailureContext, FailureVerdict
from langchain_evenkeel.jev import JevClassifier, describe_failure


class FakeClient:
    """Stands in for TypeSafeClient: records requests, answers with a fixed choice and confidence."""

    def __init__(self, choice: str, confidence: float, error: Exception | None = None) -> None:
        self.choice, self.confidence, self.error = choice, confidence, error
        self.requests: list[dict] = []

    def system_one(self, **request):
        self.requests.append(request)
        if self.error:
            raise self.error
        answer = SimpleNamespace(choice=self.choice, confidence=self.confidence)
        return SimpleNamespace(answers={"failure_kind": answer})


class FakeAsyncClient(FakeClient):
    async def system_one(self, **request):
        return FakeClient.system_one(self, **request)


def ctx(error: BaseException, **extra) -> FailureContext:
    return FailureContext(
        error=error,
        task="research-agent",
        attempt=1,
        max_attempts=3,
        payload={"topic": "durable execution"},
        **extra,
    )


def test_explicit_error_class_is_classified_by_rules_without_calling_jev():
    client = FakeClient("transient", 1.0)
    verdict = JevClassifier(client).classify(ctx(BadOutputError('unknown tool "serch_web"')))
    assert verdict == FailureVerdict("bad_output", 1.0, "rules")
    assert client.requests == []


def test_ambiguous_failure_is_classified_by_jev_from_the_error_and_context():
    client = FakeClient("bad_output", 0.87)
    verdict = JevClassifier(client).classify(
        ctx(Exception("Model requested tool `serch_web`, which is not in the provided tool list"))
    )
    assert verdict == FailureVerdict("bad_output", 0.87, "jev")
    sent = json.dumps(client.requests[0]["state"])
    assert "serch_web" in sent and "research-agent" in sent
    assert list(client.requests[0]["questions"]) == ["failure_kind"]


def test_low_confidence_falls_back_to_rules():
    rate_limited = Exception("Rate limit reached")
    rate_limited.status_code = 429
    verdict = JevClassifier(FakeClient("fatal", 0.25), min_confidence=0.5).classify(ctx(rate_limited))
    assert verdict == FailureVerdict("transient", 0.9, "rules (jev unsure)")


def test_unavailable_jev_falls_back_to_rules():
    client = FakeClient("fatal", 1.0, error=RuntimeError("TypeSafe is overloaded"))
    verdict = JevClassifier(client).classify(ctx(TypeError("'NoneType' object is not subscriptable")))
    assert verdict == FailureVerdict("fatal", 0.5, "rules (jev unavailable)")


def test_unknown_answer_falls_back_to_rules():
    verdict = JevClassifier(FakeClient("over_budget", 0.99)).classify(ctx(TypeError("bug")))
    assert verdict.source == "rules (jev answer unusable)"


def test_the_cause_and_the_model_output_are_sent():
    error = RuntimeError("request failed")
    error.__cause__ = ConnectionRefusedError("connect ECONNREFUSED 10.0.0.5:443")
    state = describe_failure(
        ctx(error, step="tool:search_web", output={"tool": "search_web", "arguments": {}})
    )
    assert "ECONNREFUSED 10.0.0.5:443" in state["error"]["cause"]
    assert state["step"] == "tool:search_web"
    assert '"tool": "search_web"' in state["output"]


def test_long_payloads_are_truncated():
    state = describe_failure(
        FailureContext(
            error=TypeError("x"), task="t", attempt=1, max_attempts=3, payload={"text": "a" * 5000}
        )
    )
    assert state["payload"].endswith("... (truncated)") and len(state["payload"]) < 2100


def test_model_is_passed_only_when_pinned():
    client = FakeClient("bad_output", 0.9)
    JevClassifier(client, model="jev-1.13.0").classify(ctx(TypeError("x")))
    JevClassifier(client).classify(ctx(TypeError("x")))
    assert client.requests[0]["model"] == "jev-1.13.0"
    assert "model" not in client.requests[1]


def test_aclassify_uses_the_async_client():
    client = FakeAsyncClient("bad_output", 0.9)
    verdict = asyncio.run(JevClassifier(async_client=client).aclassify(ctx(TypeError("x"))))
    assert verdict.source == "jev" and len(client.requests) == 1


def test_a_client_is_required():
    with pytest.raises(ValueError, match="client"):
        JevClassifier()


def test_reads_a_real_typesafe_sdk_response():
    from typesafe_sdk import SystemOneResponse

    class RealShapeClient(FakeClient):
        def system_one(self, **request):
            self.requests.append(request)
            return SystemOneResponse.model_validate(
                {
                    "model": "jev-1.13.0",
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                    "answers": {
                        "failure_kind": {
                            "type": "choice",
                            "choice": "bad_output",
                            "confidence": 0.87,
                            "probabilities": {"bad_output": 0.9, "fatal": 0.1},
                        }
                    },
                }
            )

    verdict = JevClassifier(RealShapeClient("", 0)).classify(ctx(TypeError("x")))
    assert verdict == FailureVerdict("bad_output", 0.87, "jev")
