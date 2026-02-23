# Problem 2: Redundant Exception Handling

**Priority:** HIGH

## Problem Description

The try/except/finally block in `_run_model_test_impl` catches all exceptions, logs them to test_metadata, then re-raises. However, if `record_model_test_properties()` itself throws in the finally block, it will mask the original test exception.

## Current Code Location

**File:** `tests/runner/test_models.py`
**Lines:** 109-239

```python
def _run_model_test_impl(...):
    try:
        # ... test logic ...
        comparison_result = tester.test()
        Evaluator._assert_on_results(comparison_result)  # Line 169 - May raise
    except Exception as e:
        captured = captured_output_fixture.readouterr()
        update_test_metadata_for_exception(test_metadata, e, ...)  # Line 174
        raise  # Line 177 - Re-raises after metadata update
    finally:
        # Lines 178-239: This runs REGARDLESS of whether exception was raised
        record_model_test_properties(...)  # If THIS throws, it masks the original exception!
```

## Issues

1. **Exception masking:** If property recording fails, original test exception is lost
2. **Redundant with pytest:** Pytest already has exception handling mechanism
3. **Multiple responsibilities:** Exception handling mixed with property recording
4. **Unclear flow:** Test status determination spread between exception handler and property recorder

## Questions to Answer

1. Is the exception catching actually needed, or can pytest handle it?
2. Should property recording happen in a separate pytest hook to avoid masking exceptions?
3. What metadata updates in the exception handler are actually used later?
4. Can we simplify this to let pytest's natural exception handling work?

## Related Code

- `update_test_metadata_for_exception()` in test_utils.py
- `record_model_test_properties()` in test_utils.py
- Pytest's exception handling hooks in conftest.py
