# Solution 2: Simplify Exception Handling

## Summary

Remove try/except/finally from test function and move all exception handling and property recording to pytest hooks. This prevents exception masking and simplifies test logic.

## Key Insight

Pytest already has robust exception handling via hooks. The try/except/finally in `_run_model_test_impl` is redundant and can cause exception masking when property recording fails.

## Implementation Approach

### 1. Simplify Test Function (test_models.py)

**Remove:**
- try/except/finally structure (lines 109-239)
- Direct calls to `record_model_test_properties()`
- Direct calls to `update_test_metadata_for_exception()`
- `captured_output_fixture.readouterr()` in exception handler

**Keep:**
- All test logic (creating tester, running test, asserting results)
- Store results on `request.node` for hook access

**Result:**
```python
def _run_model_test_impl(...):
    # Setup
    loader = ModelLoader(variant=variant)
    model_info = ModelLoader.get_model_info(variant=variant)

    # Store metadata for hooks
    request.node.test_metadata = test_metadata
    request.node.model_info = model_info
    request.node.run_mode = run_mode

    # Test execution (no try/except!)
    if test_metadata.status != ModelTestStatus.NOT_SUPPORTED_SKIP:
        tester = create_tester(...)
        comparison_result = tester.test(request=request)
        succeeded = all(result.passed for result in comparison_result)

        # Store for hooks
        request.node.comparison_result = comparison_result
        request.node.succeeded = succeeded

        # Let assertion raise naturally - pytest catches it
        Evaluator._assert_on_results(comparison_result)

    # No finally block!
```

### 2. Handle Exceptions in Hook (conftest.py)

```python
@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()

    if report.when != "call":
        return

    # Get metadata
    test_metadata = getattr(item, "test_metadata", None)
    if not test_metadata:
        return

    # Handle exception metadata if test failed
    if report.failed and call.excinfo:
        exc = call.excinfo.value
        # Get captured output from pytest's report
        stdout = report.capstdout if hasattr(report, "capstdout") else ""
        stderr = report.capstderr if hasattr(report, "capstderr") else ""

        # Update metadata (safe - won't mask exception)
        try:
            update_test_metadata_for_exception(
                test_metadata, exc, stdout=stdout, stderr=stderr
            )
        except Exception as e:
            logger.warning(f"Failed to update test metadata: {e}")

    # Record properties (safe - won't mask exception)
    try:
        record_model_test_properties_via_hook(item, ...)
    except Exception as e:
        logger.warning(f"Failed to record properties: {e}")
```

### 3. Prevent Exception Masking

**Key principle:** All hook code wrapped in try/except that logs but doesn't raise.

This ensures:
- Property recording failures never mask test exceptions
- Test status determined solely by test logic
- Diagnostic logs if property recording fails

### 4. Handle Output Capture

**Old:** `captured_output_fixture.readouterr()` in exception handler

**New:** Use pytest's built-in capture from report:
- `report.capstdout`: Captured stdout
- `report.capstderr`: Captured stderr

## Benefits

✅ No exception masking - property failures can't hide test failures
✅ Simpler test logic - focuses only on test execution
✅ Works with --forked - hooks run in parent process
✅ Better separation of concerns - reporting separate from test logic
✅ Pytest-native approach - uses built-in mechanisms

## Code Reduction

- ~120 lines removed from test function
- ~80 lines added to hooks
- Net: 40 line reduction + cleaner separation

## Files Modified

1. `tests/runner/test_models.py` - Remove try/except/finally (lines 109-239)
2. `tests/runner/conftest.py` - Add exception handling to hook
3. `tests/runner/test_utils.py` - Add `record_model_test_properties_via_hook()`

## Testing

1. Unit tests: Test hook functions in isolation
2. Integration tests: Run model tests with new hooks
3. Failure scenarios: Test with intentional failures
4. Regression: Compare XML output before/after
