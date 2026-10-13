"""typesafe-sdk shape that JevClassifier relies on. Offline: nothing here calls the API."""

import inspect

from typesafe_sdk import AsyncTypeSafeClient, Choice, SystemOneResponse, TypeSafeClient


def test_choice_accepts_described_options():
    Choice(instructions="Which?", criteria={"a": "first option", "b": "second option"})


def test_system_one_takes_state_questions_and_a_model():
    for client in (TypeSafeClient, AsyncTypeSafeClient):
        params = inspect.signature(client.system_one).parameters
        assert {"state", "questions", "model"} <= set(params)


def test_choice_answers_carry_the_choice_and_its_confidence():
    response = SystemOneResponse.model_validate({
        "model": "jev-1.13.0",
        "usage": {"input_tokens": 1, "output_tokens": 1},
        "answers": {"failure_kind": {"type": "choice", "choice": "bad_output", "confidence": 0.87,
                                     "probabilities": {"bad_output": 0.9, "fatal": 0.1}}},
    })
    answer = response.answers["failure_kind"]
    assert (answer.choice, answer.confidence) == ("bad_output", 0.87)
