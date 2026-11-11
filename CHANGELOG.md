# Changelog

All notable changes to this project are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.2] - 2026-11-11

### Fixed
- A reply that was paid for and then rejected now counts toward `evenkeel_spent_usd`. Before, only the reply
  that was finally accepted was counted, so a model call retried after bad output cost nothing to the budget.
  evenkeel reads the rejected reply from LangChain's structured output errors (`ai_message`) or from a
  `BadOutputError` raised with the reply as `output`. A reply it can't price warns instead of hiding the error.
- LangChain's `StructuredOutputValidationError`, `MultipleStructuredOutputsError` and `OutputParserException`
  are classified `bad_output` by the rule classifier, so they are retried with a hint instead of failing the
  run as `fatal`.

## [0.2.1] - 2026-11-10

### Added
- `EvenKeelWarning`: with `check_before_call=True`, a call evenkeel can't estimate (no output cap, no model
  name, no price) now warns once per reason instead of running unchecked in silence.
- `assumed_max_output_tokens` on `EvenKeelMiddleware`: the output cap `check_before_call` assumes for a model
  that sets none. The model's own cap wins.

### Changed
- The demo turns on `check_before_call`, so its budget moment pauses before the call that could cross the
  budget, instead of after the overshoot.

## [0.2.0] - 2026-11-10

### Added
- Python 3.14 support: CI now tests it, and the package lists it.
- An offline demo, `examples/demo.py`: one agent hits a rate limit, bad tool input, bad output, a refund that
  needs a person and an exhausted budget, with no API key. `tests/test_demo.py` checks every moment.
- `check_before_call=True` on `EvenKeelMiddleware`: before each model call, estimate the most it can cost
  (output cap plus input) and pause when that would cross the budget. Off by default; a call that can't be
  estimated runs as before. The pause payload adds `next_call_max_usd`.

### Changed
- **Requires Python 3.11 or newer.** On Python 3.10, any pause inside an async run (`ainvoke` hitting an
  escalation or the budget) failed with `RuntimeError: Called get_config outside of a runnable context`:
  before 3.11, LangGraph's `interrupt()` can't find its run inside async code. Python 3.10 reached its end of
  life in October 2026. Tests now cover pausing and resuming in async runs.

## [0.1.1] - 2026-10-14

### Fixed
- Price keys match only an exact model name or that name plus a date or version suffix (`-2024-07-18`,
  `-20241022`, `-0613`, `-001`, `-latest`). Plain prefix matching let a different model borrow a price
  (`gpt-4` priced `gpt-4o-mini` and `gpt-4-turbo`; `claude-3` priced `claude-3-5-sonnet`).

### Documentation
- README: measured classifier accuracy on keel's error sets, rules and `JevClassifier`.

## [0.1.0] - 2026-10-13

First release.

### Added
- `EvenKeelMiddleware` for LangChain agents (`create_agent`): classifies failed model and tool calls and
  backs off, retries with a hint, fails, or escalates to a person with `interrupt()`.
- Dollar budgets from token usage and a user-supplied `PriceTable`, with a pause for approval when over budget.
- Attempt records in agent state (`evenkeel_attempts`).
- `RuleClassifier` and `JevClassifier` (optional `jev` extra), ported from keel.
- Contract tests for the LangChain, LangGraph and typesafe-sdk behaviour the middleware relies on.

[Unreleased]: https://github.com/yodablocks/evenkeel/compare/v0.2.2...HEAD
[0.2.2]: https://github.com/yodablocks/evenkeel/releases/tag/v0.2.2
[0.2.1]: https://github.com/yodablocks/evenkeel/releases/tag/v0.2.1
[0.2.0]: https://github.com/yodablocks/evenkeel/releases/tag/v0.2.0
[0.1.1]: https://github.com/yodablocks/evenkeel/releases/tag/v0.1.1
[0.1.0]: https://github.com/yodablocks/evenkeel/releases/tag/v0.1.0
