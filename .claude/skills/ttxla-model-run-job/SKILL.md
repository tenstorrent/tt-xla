# TT-XLA Model Run Job

Use this skill when the user wants Claude to prepare, launch, validate, or report a bounded model-run job in `tt-xla`.

## Workflow

1. Read `docs/protocols/model-run-job-protocol.md`.
2. Select the supported job type: `nvidia_validation`, `tt_model_validation`, or `collectability_check`.
3. Ground the request in a manifest, repo ref, runner environment, and evidence destination.
4. Follow the command shapes, normalization rules, and reporting restrictions from the protocol.
5. Do not post GitHub comments or publish reports unless the user explicitly approves the exact draft.

## Output

- selected job type
- manifest path and source SHA-256
- repo ref and runner environment
- exact command or tranche plan
- evidence paths expected
- blocked preconditions, if any
