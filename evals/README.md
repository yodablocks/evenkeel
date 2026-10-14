# Classifier evaluation (opt-in)

Scores the rule classifier and the Jev cascade on keel's labelled error sets, the same way keel's README reports
them ("right kind" and "right action"). It calls the TypeSafe API, so it needs `TYPESAFE_API_KEY` and is never
run in CI. Results go to `evals/results/` (git-ignored); copy the summary lines into the main README by hand.

    uv run python evals/jev_eval.py ~/Coding_2026/keel/eval-sets/public-issues.json --model jev-1.13.0
