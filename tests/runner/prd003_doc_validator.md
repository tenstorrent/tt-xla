# PRD-003 Docs Validator

The PRD-003 validator is a pytest entrypoint for the TT-XLA documentation
installation harness. It emits contract records for the Phase 1 documentation
flows and defaults to fail-closed behavior when a command is ambiguous.

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
- `TT_XLA_PRD003_EXECUTE`: set to `1` only for an explicitly prepared VM run.
- `TT_XLA_PRD003_REQUIRE_PASS`: set to `1` when the docs are expected to be
  executable end to end; default behavior expects the current fail-closed
  documentation-gap baseline.

## Outputs

- `records/jsonl/results.jsonl`
- `records/by-flow/<flow-id>.result.json`
- `summaries/coverage-summary.json`
