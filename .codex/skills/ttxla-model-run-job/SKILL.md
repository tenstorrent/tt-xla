# TT-XLA Model Run Job

Use this skill when the user wants Codex to prepare, launch, validate, or report a bounded model-run job in `tt-xla`.

## Workflow

1. Read `docs/protocols/model-run-job-protocol.md`.
2. Select the supported job type: `nvidia_validation`, `tt_model_validation`, or `collectability_check`.
3. Verify manifest provenance, repo ref, runner environment, and evidence destination before execution.
4. Use the protocol's command shapes and outcome normalization rules.
5. Keep GitHub comments, issue updates, and stakeholder reports behind explicit user review.

## Output

- selected job type
- manifest path and source SHA-256
- repo ref and runner environment
- exact command or tranche plan
- evidence paths expected
- blocked preconditions, if any
