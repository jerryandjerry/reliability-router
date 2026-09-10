# Contributing

1. Create a focused branch.
2. Add or update tests for behavior changes.
3. Add evaluation cases for routing-policy changes.
4. Run `pytest` and `ruff check src tests`.
5. Document policy-version changes and expected metric shifts.

A pull request that changes thresholds, weights, reason codes, or hard overrides should include:

- Why the change is needed.
- New positive and negative examples.
- Before/after evaluation report.
- Expected cost and escalation-rate effect.
- Rollback plan.

Never include API keys, sensitive user data, or proprietary evaluation examples without authorization.
