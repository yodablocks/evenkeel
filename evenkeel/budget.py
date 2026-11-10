"""Turns a model response's token usage into dollars, from prices the user supplies."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage


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


# What may follow a priced name: only a date or version number (-2024-07-18, -20241022, -0613, -001) or the
# -latest alias. Anything else (-5-sonnet, -turbo, -1106-preview) is a different model with its own price.
_VERSION_SUFFIX = re.compile(r"-(\d{4}-\d{2}-\d{2}|\d{3,8}|latest)")


class PriceTable:
    """Prices per model name. A name with a version or date suffix uses its base price
    (gpt-4o-mini-2024-07-18 -> gpt-4o-mini); a different model never borrows another's price
    (gpt-4o-mini does not use gpt-4, and gpt-4-turbo does not use gpt-4)."""

    def __init__(self, prices: Mapping[str, ModelPrice]) -> None:
        self._prices = dict(prices)

    def price_for(self, model_name: str) -> ModelPrice:
        if model_name in self._prices:
            return self._prices[model_name]
        matches = [
            key
            for key in self._prices
            if model_name.startswith(key) and _VERSION_SUFFIX.fullmatch(model_name[len(key) :])
        ]
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


# Estimating a call before it runs (check_before_call=True). An estimate, not a bound: providers add tokens
# for tool formatting, and middleware can change the model, prompt or tools between calls.

_MESSAGE_OVERHEAD_TOKENS = 10
"""Tokens assumed per message for roles and separators."""


def _text_bytes(value: Any) -> int:
    return len(json.dumps(value, ensure_ascii=False, default=str).encode())


def _message_bytes(message: BaseMessage) -> int:
    size = _text_bytes(message.content)
    if isinstance(message, AIMessage) and message.tool_calls:
        size += _text_bytes(message.tool_calls)
    return size


def estimate_input_tokens(messages: Sequence[BaseMessage], *, fixed_bytes: int = 0) -> int:
    """Input tokens of the next call. From the newest reply that reported usage: its input and output tokens,
    plus the messages after it. Before any usage: every message, plus `fixed_bytes` (system prompt and tool
    schemas, which later replies' input tokens already include). Text is counted at one token per UTF-8 byte,
    which overcounts: tokens usually span several bytes."""
    start, tokens = 0, fixed_bytes
    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        usage = message.usage_metadata if isinstance(message, AIMessage) else None
        if usage:
            start, tokens = index + 1, usage["input_tokens"] + usage["output_tokens"]
            break
    return tokens + sum(_message_bytes(m) + _MESSAGE_OVERHEAD_TOKENS for m in messages[start:])


_OUTPUT_CAP_FIELDS = ("max_tokens", "max_completion_tokens", "max_output_tokens")


def _positive_int(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else None


def max_output_tokens(model: Any, settings: Mapping[str, Any]) -> int | None:
    """The call's output cap: from the call's settings, then from the model. None when nothing caps it."""
    for source in (settings, model.__dict__ if hasattr(model, "__dict__") else {}):
        for name in _OUTPUT_CAP_FIELDS:
            cap = _positive_int(source.get(name))
            if cap is not None:
                return cap
    return None


def chat_model_name(model: Any) -> str | None:
    """The model a chat model will call (ChatOpenAI.model_name, ChatAnthropic.model)."""
    for attr in ("model_name", "model"):
        name = getattr(model, attr, None)
        if isinstance(name, str) and name:
            return name
    return None
