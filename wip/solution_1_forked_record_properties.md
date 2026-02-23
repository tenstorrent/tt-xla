# Solution 1: Fix record_properties for pytest --forked

## Summary

Move property recording from finally block to pytest_runtest_makereport hook that runs in parent process, ensuring properties are recorded even when child process crashes.

## Implementation Approach

### Phase-Based Property Recording

**Two phases:**
1. **Pre-Test**: Store all known metadata on `request.node` before test runs
2. **Post-Test**: Retrieve stored metadata in pytest hook and record properties

### Key Changes

#### 1. Store Metadata (test_models.py, before try block)

```python
# Store metadata on request.node BEFORE test runs (survives crash)
request.node._test_metadata_for_report = test_metadata
request.node._model_info_for_report = model_info
request.node._run_mode_for_report = run_mode
request.node._run_phase_for_report = run_phase
request.node._parallelism_for_report = parallelism
request.node._framework_for_report = framework

# Runtime results (updated incrementally during test)
request.node._comparison_results_for_report = []
request.node._comparison_config_for_report = None
request.node._model_size_for_report = None
request.node._test_succeeded_for_report = False
```

#### 2. Update Results During Test

```python
# After tester.test() succeeds
request.node._comparison_results_for_report = list(comparison_result) if comparison_result else []
request.node._comparison_config_for_report = tester._comparison_config if tester else None
if framework == Framework.TORCH and tester is not None:
    request.node._model_size_for_report = getattr(tester, "_model_size", None)
request.node._test_succeeded_for_report = succeeded
```

#### 3. Add pytest_runtest_makereport Hook (conftest.py)

```python
@pytest.hookimpl(hookwrapper=True, trylast=True)
def pytest_runtest_makereport(item, call):
    """Record properties in parent process (survives --forked crashes)."""
    outcome = yield
    report = outcome.get_result()

    if report.when != "call":
        return

    # Get stored metadata
    test_metadata = getattr(item, "_test_metadata_for_report", None)
    if test_metadata is None:
        return

    # Record properties to report
    record_model_test_properties_to_report(report, item, ...)
```

#### 4. New Recording Function (test_utils.py)

```python
def record_model_test_properties_to_report(report, item, ...):
    """Record properties to report object (works in hooks)."""
    def mock_record_property(key, value):
        report.user_properties.append((key, value))

    # Reuse existing logic with mock fixture
    record_model_test_properties(
        record_property=mock_record_property,
        ...
    )
```

## Benefits

✅ Properties recorded even when child process crashes
✅ Backward compatible (works with/without --forked)
✅ Reuses existing property recording logic
✅ Incremental updates during test execution

## Files Modified

1. `tests/runner/test_models.py` - Add metadata storage, remove finally block recording
2. `tests/runner/conftest.py` - Add pytest_runtest_makereport hook
3. `tests/runner/test_utils.py` - Add record_model_test_properties_to_report function

## Testing

```bash
# Test with crash simulation
pytest wip/test_forked_abort_repro.py::test_simulated_abort --forked --junit-xml=crash.xml
grep "property" crash.xml  # Should see properties

# Test with real model test
pytest tests/runner/test_models.py::test_all_models_torch[...] --forked --junit-xml=model.xml
grep "model_name" model.xml  # Should see all properties
```
