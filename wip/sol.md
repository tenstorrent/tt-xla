# Solution: Record Properties with pytest --forked

## Problem

When tests run with `pytest --forked` and abort (segfault, SIGTERM, os._exit), Python's `finally` block doesn't execute in the child process. This means `record_model_test_properties` at line 191 in `tests/runner/test_models.py::_run_model_test_impl` never gets called, resulting in missing test metadata in reports.

## Root Cause

- `--forked` runs tests in a child process
- When child process is killed (segfault, signal, os._exit), it terminates immediately
- Finally blocks only run on normal Python exceptions
- The finally block code is lost with the process

## Solution

**Move `record_model_test_properties` call from the finally block to a pytest hook that runs in the PARENT process.**

### Where to Move It

Add this hook to `/Users/ndrakulic/central/repos/tt-xla/tests/runner/conftest.py`:

```python
@pytest.hookimpl(hookwrapper=True, trylast=True)
def pytest_runtest_makereport(item, call):
    """
    Record test properties in parent process (survives when --forked child crashes).

    This hook runs AFTER test completes/crashes and can add properties to the report
    that will appear in XML output.
    """
    outcome = yield
    report = outcome.get_result()

    # Only process the 'call' phase (actual test execution) or crashes
    crashed = hasattr(report, "longrepr") and report.longrepr and "CRASHED" in str(report.longrepr)

    if report.when != "call" and not crashed:
        return

    # Get test metadata that was stored on the item before test ran
    test_metadata = getattr(item, "_test_metadata_for_report", None)
    model_info = getattr(item, "_model_info_for_report", None)
    comparison_results = getattr(item, "_comparison_results_for_report", [])

    if not test_metadata:
        return  # No metadata to record

    # Initialize user_properties if not present
    if not hasattr(report, "user_properties"):
        report.user_properties = []

    # Determine test status
    test_passed = report.passed
    if crashed:
        test_passed = False
        # Extract crash signal if available
        import re
        match = re.search(r"signal (\d+)", str(report.longrepr))
        if match:
            report.user_properties.append(("crash_signal", match.group(1)))

    # Call the existing record function to add properties to the report
    # NOTE: Need to modify record_model_test_properties to accept report object
    # instead of record_property fixture
    from tests.runner.test_utils import record_model_test_properties_to_report

    record_model_test_properties_to_report(
        report=report,
        request=item._request,  # Store this in step 2 below
        model_info=model_info,
        test_metadata=test_metadata,
        run_mode=getattr(item, "_run_mode_for_report", None),
        run_phase=getattr(item, "_run_phase_for_report", None),
        parallelism=getattr(item, "_parallelism_for_report", None),
        test_passed=test_passed,
        comparison_results=comparison_results,
        comparison_config=getattr(item, "_comparison_config_for_report", None),
        model_size=getattr(item, "_model_size_for_report", None),
    )
```

### Changes to `_run_model_test_impl`

**Step 1**: Store metadata on the item BEFORE the try block:

```python
def _run_model_test_impl(...):
    # BEFORE try block - store metadata for parent process hooks
    request.node._test_metadata_for_report = test_metadata
    request.node._model_info_for_report = model_info
    request.node._run_mode_for_report = run_mode
    request.node._run_phase_for_report = run_phase
    request.node._parallelism_for_report = parallelism
    request.node._comparison_results_for_report = []
    request.node._comparison_config_for_report = None
    request.node._model_size_for_report = None
    request.node._request = request  # Store for hook access

    # ... existing code ...

    try:
        # ... existing test code ...

        comparison_result = tester.test()

        # Update stored results (will be used by hook even if test crashes)
        request.node._comparison_results_for_report = list(comparison_result) if comparison_result else []
        request.node._comparison_config_for_report = tester._comparison_config if tester else None

        if framework == Framework.TORCH and tester is not None:
            request.node._model_size_for_report = getattr(tester, "_model_size", None)

        # ... rest of existing test code ...

    except Exception as e:
        # ... existing exception handling ...
        raise
    finally:
        # REMOVE record_model_test_properties from here!
        # It now runs in the pytest_runtest_makereport hook instead
        pass
```

**Step 2**: Modify `record_model_test_properties` to work with report object:

In `tests/runner/test_utils.py`, create a new function (or modify existing):

```python
def record_model_test_properties_to_report(
    report,
    request,
    model_info,
    test_metadata,
    run_mode,
    run_phase,
    parallelism,
    test_passed,
    comparison_results,
    comparison_config,
    model_size,
):
    """
    Record test properties to a report object (for use in hooks).

    This is similar to record_model_test_properties but adds properties
    to report.user_properties instead of calling record_property fixture.
    """
    # Initialize user_properties if not present
    if not hasattr(report, "user_properties"):
        report.user_properties = []

    # Add all properties to report.user_properties
    # (Convert existing record_property calls to report.user_properties.append)

    report.user_properties.append(("model_name", model_info.name))
    report.user_properties.append(("test_passed", str(test_passed)))
    report.user_properties.append(("run_mode", str(run_mode)))
    # ... add all other properties ...
```

## Testing the Solution

1. Run the reproduction test to verify properties are recorded:

```bash
cd /Users/ndrakulic/central/repos/tt-xla
source /Users/ndrakulic/central/env/bin/activate

# Test normal exception (should work before and after)
pytest test_forked_abort_repro.py::test_normal_exception --forked --junit-xml=normal.xml
grep "property" normal.xml

# Test abort (currently fails, should work after fix)
pytest test_forked_abort_repro.py::test_simulated_abort --forked --junit-xml=abort.xml
grep "property" abort.xml
```

2. After implementing the solution, run real model tests:

```bash
pytest tests/runner/test_models.py::<some_test> --forked --junit-xml=report.xml
# Check that properties are recorded even for crashed tests
grep "property" report.xml
```

## Summary

| Location | Current (Broken) | Solution (Fixed) |
|----------|-----------------|------------------|
| Where properties recorded | Finally block in child process | Hook in parent process |
| When it runs | Only on normal exceptions | Always, even on crash |
| Data source | Local variables in try/finally | Stored on item.node before test |
| Works with --forked crashes | ❌ No | ✅ Yes |

## Key Insight

The finally block runs in the **child process** (dies on abort), but pytest hooks run in the **parent process** (survives). By storing metadata on the `item` object before the test runs, we can access it in the parent process hook after the child crashes.
