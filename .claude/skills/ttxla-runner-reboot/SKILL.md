---
name: ttxla-runner-reboot
description: Recover a blocked tt-xla model-run runner by applying the repository-owned runner reboot protocol with preflight checks, reboot evidence, and post-reboot reconciliation.
---

# TT-XLA Runner Reboot

Use this skill when the user wants Claude to reboot or recover a blocked `tt-xla` model-run runner.

## Protocol Source

The authoritative workflow is `docs/protocols/model-run-job-protocol.md`. Read it first and follow the Runner Reboot Recovery section and `docs/protocols/model-run-job-reboot-record-template.md`.

Do not fork or restate launch logic or reboot logic in this Claude wrapper. If the protocol and this wrapper disagree, follow the protocol and update this wrapper to point back to it.

## Output

Return the protocol-required reboot record path, preflight evidence, command evidence, post-reboot poll evidence, resume condition, and any human-review gates.
