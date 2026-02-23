# Focused Plan: Fix pytest --forked Property Recording

## Context

When tests run with `pytest --forked` and crash (segfault, SIGTERM, os._exit), the Python `finally` block doesn't execute in the child process. This means `record_model_test_properties()` at line 191 in `tests/runner/test_models.py` never gets called, resulting in **missing test metadata in reports**.

This is a critical issue for test observability - when tests crash, we lose all diagnostic information about which test ran, what model was being tested, and what stage it reached before crashing.

**Primary Goal:** Ensure all test properties are recorded even when tests crash with `pytest --forked`.

**Natural Piggyback:** While adding the hook for property recording, we can also move exception handling there (Problem 2), eliminating redundant try/except and preventing exception masking.

---

## Core Problem: pytest --forked Property Loss

### Root Cause

```
PARENT PROCESS (pytest)           CHILD PROCESS (--forked)
├── Collection phase
├── pytest hooks (survive crash)  ├── Test setup
│   └── pytest_runtest_makereport ├── Test execution (try block)
└── ✅ Always runs                 ├── Exception handling (except)
                                  ├── Finally block ❌
                                  │   └── record_model_test_properties()
                                  └── [CRASH - finally never runs]
```

The `pytest_runtest_makereport` hook runs in the **parent process** and ALWAYS executes, even when the child crashes. By storing metadata on the test item before execution and recording it in the hook, we guarantee properties are captured.

---

## Solution: Hook-Based Property Recording

### Phase-Based Approach

**Phase 1: Pre-Test Storage** (Before try block)
- Store all known metadata on `request.node`
- This data survives in parent process even when child crashes

**Phase 2: Incremental Updates** (During test)
- Update runtime results as they become available
- Stored on same `request.node` attributes

**Phase 3: Hook Recording** (In parent process)
- `pytest_runtest_makereport` retrieves stored metadata
- Records all properties to report
- Handles exceptions safely without masking

### Implementation Details

#### 1. Add TestReportMetadata Dataclass (test_utils.py)

Add after imports (around line 30):

```python
from dataclasses import dataclass, field
from typing import Optional, List

@dataclass
class TestReportMetadata:
    """
    Metadata container stored on pytest item for hook-based property recording.

    Survives child process crashes when using pytest --forked since it's stored
    on the item in the parent process.
    """
    # Required metadata (set before test execution)
    test_metadata: 'ModelTestConfig'
    model_info: 'ModelInfo'
    run_mode: 'RunMode'
    run_phase: 'RunPhase'
    parallelism: 'Parallelism'
    framework: 'Framework'

    # Runtime results (updated during test execution)
    comparison_results: List = field(default_factory=list)
    comparison_config: Optional = None
    model_size: Optional[int] = None
    test_succeeded: bool = False
```

#### 2. Store Metadata Before Test (test_models.py)

Add BEFORE the try block (after line 92):

```python
def _run_model_test_impl(...):
    # Setup
    fix_venv_isolation()
    loader_path = test_entry.path
    variant, ModelLoader = test_entry.variant_info

    with RequirementsManager.for_loader(loader_path):
        loader = ModelLoader(variant=variant)
        model_info = ModelLoader.get_model_info(variant=variant)

        # =====================================================
        # STORE METADATA ON NODE (survives child crash)
        # =====================================================
        from tests.runner.test_utils import TestReportMetadata

        request.node._report_metadata = TestReportMetadata(
            test_metadata=test_metadata,
            model_info=model_info,
            run_mode=run_mode,
            run_phase=run_phase,
            parallelism=parallelism,
            framework=framework,
        )
        # =====================================================

        # ... existing IR dump setup code ...

        try:
            # ... existing test execution ...
```

#### 3. Update Results During Test (test_models.py)

Add AFTER test execution (after line 162):

```python
            # Run test
            comparison_result = tester.test(request=request)

            # =====================================================
            # UPDATE STORED RESULTS (incremental)
            # =====================================================
            metadata = request.node._report_metadata
            metadata.comparison_results = (
                list(comparison_result) if comparison_result else []
            )
            metadata.comparison_config = (
                tester._comparison_config if tester else None
            )
            if framework == Framework.TORCH and tester is not None:
                metadata.model_size = getattr(tester, "_model_size", None)
            # =====================================================

            # All results must pass
            succeeded = all(result.passed for result in comparison_result)

            # =====================================================
            # UPDATE SUCCESS STATUS
            # =====================================================
            metadata.test_succeeded = succeeded
            # =====================================================

            # Assert on results
            Evaluator._assert_on_results(comparison_result)
```

#### 4. Update Metadata on Exception (test_models.py)

Modify exception handler (after line 174):

```python
        except Exception as e:
            captured = captured_output_fixture.readouterr()
            update_test_metadata_for_exception(
                test_metadata, e, stdout=captured.out, stderr=captured.err
            )
            # =====================================================
            # UPDATE STORED METADATA (survives crash after this)
            # =====================================================
            # Metadata object is already mutated in place, but ensure
            # test_metadata reference is updated
            request.node._report_metadata.test_metadata = test_metadata
            # =====================================================
            raise
```

#### 5. Remove Property Recording from Finally Block (test_models.py)

Replace lines 186-198 with comment:

```python
        finally:
            # =====================================================
            # REMOVED: record_model_test_properties call
            # Now handled by pytest_runtest_makereport hook
            # Properties are recorded in parent process, ensuring
            # they survive even when child process crashes with --forked
            # =====================================================

            # Keep perf benchmark reporting (existing code)
            if framework == Framework.TORCH and run_mode == RunMode.INFERENCE:
                # ... existing benchmark code ...
```

#### 6. Add pytest_runtest_makereport Hook (conftest.py)

Add at end of file (after line 179):

```python
@pytest.hookimpl(hookwrapper=True, trylast=True)
def pytest_runtest_makereport(item, call):
    """
    Record test properties in parent process.

    This hook runs AFTER test completes or crashes and adds properties to the
    report. It runs in the parent process (not the forked child), so it ALWAYS
    executes even when the child process is killed.

    This solves the pytest --forked crash problem where finally blocks in the
    child process never run when the process is killed.
    """
    outcome = yield
    report = outcome.get_result()

    # Only process the 'call' phase (actual test execution)
    if report.when != "call":
        return

    # Get stored metadata - check if this is a model test
    metadata = getattr(item, "_report_metadata", None)
    if metadata is None:
        return  # Not a model test or metadata not stored

    # Determine test pass/fail status from report
    test_passed = report.passed

    # Handle exception metadata if test failed
    # PIGGYBACK: Move exception handling to hook (Problem 2)
    if report.failed and call.excinfo:
        exc = call.excinfo.value
        # Get captured output from pytest's report
        stdout = report.capstdout if hasattr(report, "capstdout") else ""
        stderr = report.capstderr if hasattr(report, "capstderr") else ""

        # Update test metadata with exception info
        # Wrapped in try/except to prevent masking original exception
        try:
            from tests.runner.test_utils import update_test_metadata_for_exception
            update_test_metadata_for_exception(
                metadata.test_metadata, exc, stdout=stdout, stderr=stderr
            )
        except Exception as meta_error:
            # Log but don't fail - property recording shouldn't break tests
            import logging
            logging.warning(
                f"Failed to update test metadata for exception: {meta_error}"
            )

    # Record properties to the report object
    # Wrapped in try/except to prevent masking test exceptions
    try:
        from tests.runner.test_utils import record_model_test_properties_to_report

        record_model_test_properties_to_report(
            report=report,
            item=item,
            model_info=metadata.model_info,
            test_metadata=metadata.test_metadata,
            run_mode=metadata.run_mode,
            run_phase=metadata.run_phase,
            parallelism=metadata.parallelism,
            test_passed=test_passed,
            comparison_results=metadata.comparison_results,
            comparison_config=metadata.comparison_config,
            model_size=metadata.model_size,
        )
    except Exception as prop_error:
        # Log but don't fail - property recording shouldn't break tests
        import logging
        logging.warning(
            f"Failed to record model test properties: {prop_error}"
        )
```

#### 7. Add record_model_test_properties_to_report (test_utils.py)

Add after existing `record_model_test_properties` function (after line 682):

```python
def record_model_test_properties_to_report(
    report,
    item,
    *,
    model_info,
    test_metadata,
    run_mode: RunMode,
    parallelism: Parallelism,
    run_phase: RunPhase = RunPhase.DEFAULT,
    test_passed: bool = False,
    comparison_results: list = None,
    comparison_config=None,
    model_size: int = None,
):
    """
    Record test properties to a pytest report object (for use in hooks).

    This is similar to record_model_test_properties() but adds properties
    to report.user_properties instead of calling the record_property fixture.
    This allows it to be called from pytest hooks that run in the parent process
    when using pytest --forked.

    Args:
        report: pytest TestReport object to add properties to
        item: pytest Item object representing the test
        model_info: ModelInfo for the model being tested
        test_metadata: ModelTestConfig for the test
        run_mode: RunMode (INFERENCE/TRAINING)
        parallelism: Parallelism mode
        run_phase: RunPhase (DEFAULT/LLM_PREFILL/LLM_DECODE)
        test_passed: Whether the test passed
        comparison_results: List of ComparisonResult objects
        comparison_config: ComparisonConfig used for the test
        model_size: Model size in parameters
    """
    # Initialize user_properties if not present
    if not hasattr(report, "user_properties"):
        report.user_properties = []

    # Create a mock record_property that appends to report.user_properties
    def mock_record_property(key, value):
        report.user_properties.append((key, value))

    # Create a mock request object with the item's node
    class MockRequest:
        def __init__(self, item):
            self.node = item

    mock_request = MockRequest(item)

    # Call the existing record_model_test_properties function
    # It will use our mock_record_property to add to report.user_properties
    # Note: This may call pytest.skip() or pytest.xfail() which will raise
    try:
        record_model_test_properties(
            record_property=mock_record_property,
            request=mock_request,
            model_info=model_info,
            test_metadata=test_metadata,
            run_mode=run_mode,
            run_phase=run_phase,
            parallelism=parallelism,
            test_passed=test_passed,
            comparison_results=comparison_results,
            comparison_config=comparison_config,
            model_size=model_size,
        )
    except (pytest.skip.Exception, pytest.xfail.Exception):
        # Skip/xfail exceptions are expected for certain test statuses
        # The report already has the correct outcome, just record properties
        pass
```

---

## Piggyback: Simplify Exception Handling (Problem 2)

Since we're already adding the `pytest_runtest_makereport` hook, we can move exception handling there too. This eliminates:
1. Redundant try/except in test function
2. Risk of exception masking if property recording fails
3. Need for `captured_output_fixture.readouterr()` in exception handler

### Changes for Problem 2 Piggyback

**In test_models.py - Remove try/except (Optional, can defer):**
- The try/except at lines 109, 171-177 can be removed
- But to minimize risk, we can keep it for now and remove in a follow-up
- The finally block removal (line 186-198) is mandatory for Problem 1

**In conftest.py hook - Already added:**
- Exception handling code is already in the hook above (lines with `if report.failed and call.excinfo`)
- Uses `report.capstdout/capstderr` instead of `captured_output_fixture`

**Benefits of this piggyback:**
- ✅ No exception masking (hook code wrapped in try/except with logging)
- ✅ Simpler flow (exception metadata updated in one place)
- ✅ Uses pytest's built-in capture mechanism
- ✅ Minimal additional code (already in the hook we're adding)

---

## Files Modified

### Must Modify:
1. **tests/runner/test_utils.py**
   - Add TestReportMetadata dataclass (after imports): ~15 lines
   - Add record_model_test_properties_to_report function (after line 682): ~60 lines
   - **Net change:** +75 lines

2. **tests/runner/test_models.py**
   - Add metadata storage before try block (after line 92): ~10 lines (much cleaner!)
   - Add incremental updates during test (after line 162): ~7 lines (much cleaner!)
   - Update exception handler to store metadata (after line 174): ~3 lines
   - Remove property recording from finally (lines 186-198): ~12 lines removed
   - **Net change:** +8 lines

3. **tests/runner/conftest.py**
   - Add pytest_runtest_makereport hook (after line 179): ~50 lines (shorter with dataclass!)
   - **Net change:** +50 lines

**Total: +133 lines (well-focused changes with improved readability)**

---

## Testing Strategy

### 1. Unit Tests (Optional but Recommended)

Create `tests/runner/test_forked_property_recording.py`:

```python
def test_properties_recorded_on_crash(tmp_path):
    """Verify properties are recorded even when test crashes."""
    # Create a test that simulates crash
    test_file = tmp_path / "test_crash.py"
    test_file.write_text('''
import os
import pytest
from tests.runner.test_utils import TestReportMetadata
from tests.runner.enums import RunMode, RunPhase, Parallelism, Framework

@pytest.mark.model_test
def test_simulated_crash(request):
    # Store metadata like real tests using dataclass
    class MockMetadata:
        status = "EXPECTED_PASSING"
        bringup_status = "UNKNOWN"

    class MockModelInfo:
        name = "test_model"

    request.node._report_metadata = TestReportMetadata(
        test_metadata=MockMetadata(),
        model_info=MockModelInfo(),
        run_mode=RunMode.INFERENCE,
        run_phase=RunPhase.DEFAULT,
        parallelism=Parallelism.SINGLE_DEVICE,
        framework=Framework.TORCH,
    )

    # Simulate crash
    os._exit(1)
''')

    # Run with --forked and capture XML
    result = pytest.main([
        str(test_file),
        '--forked',
        '--junit-xml=crash.xml'
    ])

    # Verify properties are in XML despite crash
    import xml.etree.ElementTree as ET
    tree = ET.parse('crash.xml')
    properties = tree.find('.//properties')
    assert properties is not None
    assert any(prop.attrib['name'] == 'model_name' for prop in properties)
```

### 2. Integration Tests (Critical)

```bash
# Test with existing model tests
pytest tests/runner/test_models.py::test_all_models_torch[specific_test] \
  --forked --junit-xml=output.xml

# Verify properties in XML
grep "property" output.xml | grep "model_name"
grep "property" output.xml | grep "bringup_status"
```

### 3. Regression Tests (Critical)

```bash
# Compare forked vs non-forked output
pytest tests/runner/test_models.py::test_all_models_torch[...] \
  --junit-xml=no_fork.xml

pytest tests/runner/test_models.py::test_all_models_torch[...] \
  --forked --junit-xml=with_fork.xml

# Properties should be identical
diff <(grep "property" no_fork.xml | sort) \
     <(grep "property" with_fork.xml | sort)
```

### 4. Crash Simulation (Critical)

Create `tests/runner/test_forked_abort_repro.py`:

```python
import os
import pytest
from tests.runner.test_utils import TestReportMetadata
from tests.runner.enums import RunMode, RunPhase, Parallelism, Framework

def test_simulated_abort(request):
    """Simulate a crash to verify property recording."""
    # Store minimal metadata using dataclass
    class MockMetadata:
        status = "EXPECTED_PASSING"
        bringup_status = "UNKNOWN"

    class MockModelInfo:
        name = 'test_model'

    request.node._report_metadata = TestReportMetadata(
        test_metadata=MockMetadata(),
        model_info=MockModelInfo(),
        run_mode=RunMode.INFERENCE,
        run_phase=RunPhase.DEFAULT,
        parallelism=Parallelism.SINGLE_DEVICE,
        framework=Framework.TORCH,
    )

    # Force crash (finally block won't run)
    os._exit(1)
```

Run with:
```bash
pytest tests/runner/test_forked_abort_repro.py::test_simulated_abort \
  --forked --junit-xml=abort.xml

# Should see properties despite crash
grep "model_name" abort.xml
```

---

## Verification Checklist

After implementation, verify:

- [ ] Properties recorded with `--forked` flag
- [ ] Properties recorded without `--forked` flag (backward compatibility)
- [ ] Properties identical in both cases
- [ ] Crash simulation test passes
- [ ] All existing model tests pass
- [ ] XML reports contain all expected properties
- [ ] Exception metadata captured correctly
- [ ] No exception masking observed
- [ ] Performance acceptable (minimal overhead)

---

## Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Properties lost if storage fails | High | Store metadata immediately after ModelLoader creation |
| Hook doesn't run | Critical | Use `hookwrapper=True, trylast=True` to ensure execution |
| Exception masking | High | Wrap all hook code in try/except with logging |
| Backward compatibility broken | Medium | Test with and without --forked; keep fixture-based recording |
| Missing metadata | Medium | Defensive getattr() with defaults |

---

## Benefits

✅ **Solves --forked crash problem** - Properties recorded even when child crashes
✅ **No exception masking** - Hook code wrapped safely
✅ **Backward compatible** - Works with and without --forked
✅ **Minimal changes** - Only 3 files modified, well-focused
✅ **Reuses existing logic** - `record_model_test_properties()` unchanged
✅ **Natural piggyback** - Exception handling moves to hook with minimal cost
✅ **Clean dataclass design** - Single `TestReportMetadata` object instead of 10+ attributes
✅ **Type safety** - Dataclass provides structure, type hints, and self-documentation
✅ **Easier to maintain** - Adding new metadata fields is trivial

---

## What's NOT in This Plan (Future Work)

These problems are deferred to separate PRs:

- **Problem 3:** Status determination simplification (BringupStatusResolver)
- **Problem 5:** Tester initialization factory pattern
- **Problem 7:** Guidance tag integration
- **Problem 8:** Marshal safety wrapper (SafePropertyRecorder)

These can be addressed after the core --forked fix is stable.

---

## Implementation Estimate

- **Code changes:** 2-3 hours
- **Testing:** 2-3 hours
- **Review/refinement:** 1-2 hours
- **Total:** 5-8 hours for a focused, well-tested fix

---

## Success Criteria

✅ All properties recorded when tests crash with `--forked`
✅ All existing tests pass without modification
✅ XML report output matches non-forked execution
✅ No exception masking observed
✅ Crash simulation test passes
