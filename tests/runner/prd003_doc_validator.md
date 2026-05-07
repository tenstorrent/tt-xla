# PRD-003 Docs and Demo Validator

The PRD-003 validator is a pytest entrypoint for the TT-XLA documentation and
demo validation harness. It emits contract records for the Phase 1
documentation/install flows and the P0/P1 hardware-sensitive demo gates.

Documentation flows default to fail-closed behavior when a command is
ambiguous. Hardware-sensitive demo gates default to explicit
`infra-unavailable` records until an IRD or equivalent TT hardware target is
proven and execution is enabled.

## Local Pytest Command

```bash
PYTHONPATH=. TT_XLA_PRD003_OUTPUT_ROOT=artifacts/prd003_doc_validator_pytest \
TT_XLA_PRD003_RUN_ID=prd003-doc-validator-pytest-$(date +%Y%m%d%H%M%S) \
pytest -q --noconftest tests/runner/test_prd003_doc_validator.py
```

Use `--noconftest` for minimal local environments where TT-XLA runtime
dependencies such as `torch` are not installed. The test itself only requires
the Python standard library and pytest.

## Environment

- `TT_XLA_PRD003_OUTPUT_ROOT`: artifact root for JSONL records and summaries.
- `TT_XLA_PRD003_RUN_ID`: run identity embedded in emitted records.
- `TT_XLA_PRD003_TARGET_ID`: target identity, defaulting to `local`.
- `TT_XLA_PRD003_EXECUTE`: set to `1` only for an explicitly prepared VM or
  IRD-equivalent TT hardware run.
- `TT_XLA_PRD003_REQUIRE_PASS`: set to `1` when the docs are expected to be
  executable end to end; default behavior expects fail-closed documentation
  gaps and explicit hardware-gate infrastructure errors.

For IRD source-build validation, the harness tracks the documented command with
`-DTT_USE_SYSTEM_SFPI=OFF` so the pinned `tt-metal` revision uses its matching
user-local SFPI toolchain instead of a potentially stale system package under
`/opt/tenstorrent/sfpi`.

## Outputs

- `records/jsonl/results.jsonl`
- `records/by-flow/<flow-id>.result.json`
- `summaries/coverage-summary.json`
