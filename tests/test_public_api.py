import evenkeel


def test_public_names_import_from_the_package_root():
    for name in (
        "EvenKeelMiddleware",
        "PriceTable",
        "ModelPrice",
        "JevClassifier",
        "RuleClassifier",
        "default_policy",
        "BadInputError",
        "BadOutputError",
        "NeedsHumanError",
        "OverBudgetError",
        "FailureVerdict",
        "FailureContext",
        "AttemptRecord",
    ):
        assert hasattr(evenkeel, name), name
