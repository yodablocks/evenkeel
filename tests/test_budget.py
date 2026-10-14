import pytest
from langchain_core.messages import AIMessage

from evenkeel.budget import MissingUsageError, ModelPrice, PriceTable, UnknownModelPriceError
from tests.fakes import ai

PRICES = PriceTable({"fake-model": ModelPrice(input_per_mtok=1000, output_per_mtok=2000)})


def test_cost_is_tokens_times_price_per_million():
    assert PRICES.cost_usd(ai("x", tokens=(100, 50))) == pytest.approx(0.2)


def test_dated_model_names_use_the_longest_matching_prefix():
    table = PriceTable({"gpt-4o": ModelPrice(2.5, 10), "gpt-4o-mini": ModelPrice(0.15, 0.6)})
    assert table.price_for("gpt-4o-mini-2024-07-18") == ModelPrice(0.15, 0.6)
    assert table.price_for("gpt-4o-2024-08-06") == ModelPrice(2.5, 10)


def test_unknown_model_is_an_error_not_zero():
    with pytest.raises(UnknownModelPriceError, match="other-model"):
        PRICES.cost_usd(ai("x", model="other-model"))


def test_missing_usage_is_an_error_not_zero():
    message = AIMessage(content="x", response_metadata={"model_name": "fake-model"})
    with pytest.raises(MissingUsageError, match="fake-model"):
        PRICES.cost_usd(message)


def test_missing_model_name_names_the_metadata_keys():
    message = AIMessage(
        content="x", usage_metadata={"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}
    )
    with pytest.raises(UnknownModelPriceError, match="response_metadata"):
        PRICES.cost_usd(message)


def test_model_key_is_read_when_model_name_is_absent():
    message = ai("x")
    message.response_metadata = {"model": "fake-model"}
    assert PRICES.cost_usd(message) == pytest.approx(0.2)


@pytest.mark.parametrize(
    ("model", "key"),
    [
        ("gpt-4o-mini-2024-07-18", "gpt-4o-mini"),
        ("claude-3-5-sonnet-20241022", "claude-3-5-sonnet"),
        ("claude-3-5-sonnet-latest", "claude-3-5-sonnet"),
        ("gpt-4-0613", "gpt-4"),
        ("gemini-1.5-pro-001", "gemini-1.5-pro"),
    ],
)
def test_version_and_date_suffixes_use_the_base_price(model, key):
    assert PriceTable({key: ModelPrice(1, 2)}).price_for(model) == ModelPrice(1, 2)


@pytest.mark.parametrize(
    ("model", "key"),
    [
        ("gpt-4o-mini", "gpt-4"),
        ("gpt-4o-mini-2024-07-18", "gpt-4o"),
        ("gpt-4-turbo", "gpt-4"),
        ("claude-3-5-sonnet-20241022", "claude-3"),
        ("gpt-4-1106-preview", "gpt-4"),
    ],
)
def test_a_different_model_never_borrows_a_price(model, key):
    with pytest.raises(UnknownModelPriceError, match=model):
        PriceTable({key: ModelPrice(1, 2)}).price_for(model)
