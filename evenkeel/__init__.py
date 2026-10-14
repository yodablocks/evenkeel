"""Failure classification, retries with hints, dollar budgets and escalation for LangChain agents."""

from ._core import AttemptRecord
from .budget import ModelPrice, PriceTable
from .failure import (
    BadInputError,
    BadOutputError,
    FailureContext,
    FailureVerdict,
    NeedsHumanError,
    OverBudgetError,
    RuleClassifier,
    default_policy,
)
from .jev import JevClassifier
from .middleware import EvenKeelMiddleware

__version__ = "0.1.0"

__all__ = [
    "AttemptRecord",
    "BadInputError",
    "BadOutputError",
    "EvenKeelMiddleware",
    "FailureContext",
    "FailureVerdict",
    "JevClassifier",
    "ModelPrice",
    "NeedsHumanError",
    "OverBudgetError",
    "PriceTable",
    "RuleClassifier",
    "default_policy",
]
