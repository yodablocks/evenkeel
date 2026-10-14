"""Turns a model response's token usage into dollars, from prices the user supplies."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from langchain_core.messages import AIMessage


class UnknownModelPriceError(ValueError):
    """The response's model is not in the price table."""


class MissingUsageError(ValueError):
    """The response reported no token usage, so its cost is unknown."""


@dataclass(frozen=True)
class ModelPrice:
    input_per_mtok: float
    """Dollars per million input tokens."""
    output_per_mtok: float
    """Dollars per million output tokens."""


def model_name_of(message: AIMessage) -> str | None:
    metadata = message.response_metadata or {}
    name = metadata.get("model_name") or metadata.get("model")
    return name if isinstance(name, str) else None


class PriceTable:
    """Prices per model name. A dated name such as gpt-4o-mini-2024-07-18 uses the longest matching key."""

    def __init__(self, prices: Mapping[str, ModelPrice]) -> None:
        self._prices = dict(prices)

    def price_for(self, model_name: str) -> ModelPrice:
        if model_name in self._prices:
            return self._prices[model_name]
        matches = [key for key in self._prices if model_name.startswith(key)]
        if not matches:
            raise UnknownModelPriceError(
                f"No price for model {model_name!r}. Add it to PriceTable; known: {sorted(self._prices)}"
            )
        return self._prices[max(matches, key=len)]

    def cost_usd(self, message: AIMessage) -> float:
        name = model_name_of(message)
        if name is None:
            raise UnknownModelPriceError(
                "The response does not say which model produced it; response_metadata keys: "
                f"{sorted((message.response_metadata or {}).keys())}"
            )
        price = self.price_for(name)
        usage = message.usage_metadata
        if not usage:
            raise MissingUsageError(f"Model {name!r} reported no token usage, so this call's cost is unknown")
        return (
            usage["input_tokens"] * price.input_per_mtok + usage["output_tokens"] * price.output_per_mtok
        ) / 1e6
