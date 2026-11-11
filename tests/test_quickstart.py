"""The README's five-minute guide runs as printed, and stays identical to examples/quickstart.py."""

import re
import runpy
import warnings
from pathlib import Path

from evenkeel import EvenKeelWarning

ROOT = Path(__file__).resolve().parent.parent
QUICKSTART = ROOT / "examples" / "quickstart.py"


def test_the_quickstart_pauses_before_the_budget_then_retries_bad_output_with_a_hint(capsys):
    with warnings.catch_warnings():
        warnings.simplefilter("error", EvenKeelWarning)  # the budget check must not be silently off
        runpy.run_path(str(QUICKSTART), run_name="__main__")
    assert capsys.readouterr().out.splitlines() == [
        "Paused: spent $0.0033 of $0.005, next call up to $0.0037",
        "Retried: StructuredOutputValidationError -> bad_output, retry_modified",
        "Done: Refund(order_id='A-1042', amount_usd=64.0), spent $0.0099 for 3 calls",
    ]


def test_the_readme_shows_the_quickstart_exactly():
    readme = (ROOT / "README.md").read_text()
    blocks = re.findall(r"```python\n(.*?)```", readme, re.S)
    assert QUICKSTART.read_text() in blocks, "copy examples/quickstart.py into the README's five-minute guide"
