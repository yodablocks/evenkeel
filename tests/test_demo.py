"""Runs examples/demo.py and checks every moment it shows, so the demo cannot silently drift from the code."""

import importlib.util
import sys
import warnings
from pathlib import Path

DEMO = Path(__file__).resolve().parent.parent / "examples" / "demo.py"


def load_demo():
    spec = importlib.util.spec_from_file_location("evenkeel_demo", DEMO)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # pydantic resolves the model's annotations through sys.modules
    spec.loader.exec_module(module)
    return module


def test_the_demo_shows_every_failure_kind_and_completes(capsys):
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # the demo must not run with check_before_call silently off
        result = load_demo().main()
    log = "\n".join(result["log"])
    state = result["state"]

    # 1. Rate limit: backed off and retried.
    assert "-> transient (confidence 0.90), so back off" in log
    # 2. Bad tool input: the error went back to the model, which then used a valid id.
    assert "-> bad_input (confidence 1.00), so hand the error back to the model" in log
    assert "lookup_order('A-1042') found" in log
    # 3. Bad output: the retry carried evenkeel's hint to the model.
    assert "with evenkeel's hint: [evenkeel] The previous attempt produced unusable output" in log
    # 4. Budget: paused before a call that could cross $0.03, while spend was still under it.
    assert (
        "[reviewer] paused before a call: spent $0.0210 of a $0.03 budget, "
        "and the next call could cost up to $0.0117" in log
    )
    assert state["evenkeel_budget_usd"] == 0.05
    # 5. Needs a person: paused, and the reviewer's skip applied after LangGraph's replay.
    assert "[reviewer] paused: A refund of $640 is over the $500 limit" in log
    assert "re-ran the paused call on resume. The decision applies now." in log
    assert log.index("paused before a call") < log.index("paused: A refund")

    assert state["messages"][-1].content.startswith("Your refund of $640 for order A-1042 is done")
    assert round(state["evenkeel_spent_usd"], 4) == 0.042  # four answered calls at $0.0105
    assert [(r["call"], r["kind"], r["action"]) for r in state["evenkeel_attempts"]] == [
        ("model", "transient", "retry"),
        ("tool:lookup_order", "bad_input", "returned_to_model"),
        ("model", "bad_output", "retry_modified"),
        ("tool:refund", "needs_human", "skipped_by_reviewer"),
    ]
    assert "[demo] spent $0.0420" in capsys.readouterr().out
