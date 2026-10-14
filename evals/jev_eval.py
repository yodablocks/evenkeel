"""Opt-in: score the rule classifier and JevClassifier on keel's labelled error sets.
Calls the TypeSafe API with TYPESAFE_API_KEY. Never runs in CI.

Usage:
  uv run python evals/jev_eval.py ~/Coding_2026/keel/eval-sets/public-issues.json
  uv run python evals/jev_eval.py ~/Coding_2026/keel/eval-sets/fresh-issues.json --model jev-1.13.0
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

from evenkeel import FailureContext, JevClassifier, RuleClassifier

# What keel's default policy does for each kind, used for the "right action" score.
ACTION_OF = {
    "transient": "retry",
    "bad_output": "retry_modified",
    "bad_input": "fail",
    "fatal": "fail",
    "needs_human": "escalate",
    "over_budget": "escalate",
}


def as_exception(error: dict) -> Exception:
    cls = type(error.get("name") or "Error", (Exception,), {})
    exc = cls(error.get("message", ""))
    if "status" in error:
        exc.status_code = error["status"]
    if "code" in error:
        exc.code = error["code"]
    if "cause" in error:
        exc.__cause__ = Exception(str(error["cause"]))
    return exc


def score(classifier, cases: list[dict]) -> dict:
    right_kind = right_action = 0
    rows = []
    for case in cases:
        ctx = FailureContext(
            error=as_exception(case["error"]), task=case.get("task", "agent"), attempt=1, max_attempts=3
        )
        verdict = classifier.classify(ctx)
        right_kind += verdict.kind == case["label"]
        right_action += ACTION_OF[verdict.kind] == ACTION_OF[case["label"]]
        rows.append(
            {
                "id": case["id"],
                "label": case["label"],
                "kind": verdict.kind,
                "confidence": verdict.confidence,
                "source": verdict.source,
            }
        )
    return {"right_kind": right_kind, "right_action": right_action, "n": len(cases), "rows": rows}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("cases", type=Path)
    parser.add_argument("--model", default=None, help="pin a Jev version, e.g. jev-1.13.0")
    args = parser.parse_args()

    cases = [c for c in json.loads(args.cases.read_text()) if c.get("label")]
    from typesafe_sdk import TypeSafeClient

    results = {
        "rules": score(RuleClassifier(), cases),
        "jev_cascade": score(JevClassifier(TypeSafeClient(), model=args.model), cases),
    }
    for name, result in results.items():
        n = result["n"]
        print(
            f"{name:12} right kind {result['right_kind']}/{n} ({result['right_kind'] / n:.0%}), "
            f"right action {result['right_action']}/{n} ({result['right_action'] / n:.0%})"
        )

    out = Path(__file__).parent / "results" / f"{args.cases.stem}-{dt.date.today().isoformat()}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"cases": str(args.cases), "model": args.model, **results}, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
