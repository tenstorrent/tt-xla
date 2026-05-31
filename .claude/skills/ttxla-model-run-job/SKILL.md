# TT-XLA Model Run Job

Use this skill when the user wants Claude to prepare, launch, validate, or report a bounded model-run job in `tt-xla`.

## Protocol Source

The authoritative workflow is `docs/protocols/model-run-job-protocol.md`. Read it first and treat it as the single source of truth for job types, preconditions, command shapes, evidence capture, normalization, and reporting restrictions.

Do not fork or restate launch logic in this Claude wrapper. If the protocol and this wrapper disagree, follow the protocol and update this wrapper to point back to it.

## Output

Return the protocol-required job plan, evidence paths, blocked preconditions, and any human-review gates.
