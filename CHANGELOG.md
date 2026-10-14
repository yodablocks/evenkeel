# Changelog

All notable changes to this project are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-10-13

First release.

### Added
- `EvenKeelMiddleware` for LangChain agents (`create_agent`): classifies failed model and tool calls and
  backs off, retries with a hint, fails, or escalates to a person with `interrupt()`.
- Dollar budgets from token usage and a user-supplied `PriceTable`, with a pause for approval when over budget.
- Attempt records in agent state (`evenkeel_attempts`).
- `RuleClassifier` and `JevClassifier` (optional `jev` extra), ported from keel.
- Contract tests for the LangChain, LangGraph and typesafe-sdk behaviour the middleware relies on.

[0.1.0]: https://github.com/yodablocks/evenkeel/releases/tag/v0.1.0
