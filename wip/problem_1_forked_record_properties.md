# Problem 1: record_properties fails with pytest --forked

**Priority:** HIGH (Original from PR_PLAN.md)

## Problem Description

When tests run with `pytest --forked` and abort (segfault, SIGTERM, os._exit), Python's `finally` block doesn't execute in the child process. This means `record_model_test_properties` at line 191 in `tests/runner/test_models.py::_run_model_test_impl` never gets called, resulting in missing test metadata in reports.

## Current Code Location

**File:** `tests/runner/test_models.py`
**Function:** `_run_model_test_impl` (lines 54-239)
**Problem line:** 191 (inside finally block)

```python
def _run_model_test_impl(...):
    try:
        # Test code runs here
        comparison_result = tester.test()
        ...
    except Exception as e:
        ...
        raise
    finally:
        # ❌ THIS RUNS IN CHILD PROCESS
        # ❌ DOES NOT RUN when child process is killed (segfault/SIGTERM/os._exit)
        record_model_test_properties(...)  # <-- LINE 191
```

## Root Cause

- `--forked` runs tests in a child process
- When child process is killed (segfault, signal, os._exit), it terminates immediately
- Finally blocks only run on normal Python exceptions
- The finally block code is lost with the process

## Requirements

1. Properties known before test starts should be recorded beforehand
2. Properties calculated during test should be recorded in test teardown
3. Must work even when child process crashes with --forked
4. Should not duplicate property recording logic

## Related Files

- `tests/runner/test_models.py` - Main test implementation
- `tests/runner/conftest.py` - Pytest hooks and fixtures
- `tests/runner/test_utils.py` - Property recording functions
