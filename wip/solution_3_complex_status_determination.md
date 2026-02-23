# Solution 3: Simplify Status Determination Logic

## Summary

Extract status determination into a dedicated `BringupStatusResolver` class with explicit resolution hierarchy. This makes the 7+ conditional branches clear, testable, and maintainable.

## Current Problem

**Multiple sources of truth:**
1. test_metadata.status (ModelTestStatus)
2. test_metadata.bringup_status (BringupStatus)
3. comparison_results (runtime)
4. parse_last_bringup_stage() (from file)
5. test_metadata.failing_reason (from exception)
6. test_passed parameter

**Result:** 7+ nested conditionals, unclear precedence, hard to test

## Solution Architecture

### 1. Create Status Resolution Class

**New file: tests/runner/status_resolver.py**

```python
class StatusResolutionContext:
    """Input data for status resolution."""
    def __init__(
        self,
        model_test_status: ModelTestStatus,
        config_bringup_status: BringupStatus | None,
        test_passed: bool,
        comparison_results: list[ComparisonResult] | None,
        comparison_config: ComparisonConfig | None,
        parsed_stage: BringupStatus | None = None,
    ):
        ...

class StatusResolution:
    """Result with explanation."""
    def __init__(
        self,
        bringup_status: BringupStatus,
        reason: str,
        source: str,  # Which rule determined this
    ):
        ...

class BringupStatusResolver:
    """
    Resolves bringup status using clear hierarchy.

    Resolution Hierarchy (first match wins):
    1. NOT_SUPPORTED_SKIP → use config_bringup_status
    2. NOT_STARTED → use config_bringup_status
    3. Has comparison_results + INCORRECT_RESULT → INCORRECT_RESULT
    4. Has comparison_results + test_passed → PASSED
    5. No comparison_results → parse_last_bringup_stage() or UNKNOWN
    """

    @staticmethod
    def resolve(context: StatusResolutionContext) -> StatusResolution:
        """Apply resolution rules in priority order."""

        # Rule 1: NOT_SUPPORTED_SKIP
        if context.model_test_status == ModelTestStatus.NOT_SUPPORTED_SKIP:
            return StatusResolution(
                bringup_status=context.config_bringup_status,
                reason="",
                source="config_not_supported"
            )

        # Rule 2: NOT_STARTED
        if context.config_bringup_status == BringupStatus.NOT_STARTED:
            return StatusResolution(
                bringup_status=BringupStatus.NOT_STARTED,
                reason="",
                source="config_not_started"
            )

        # Rule 3 & 4: Runtime comparison results
        if context.comparison_results:
            return BringupStatusResolver._resolve_from_comparison(context)

        # Rule 5: No comparison results (failure)
        return BringupStatusResolver._resolve_from_failure(context)
```

### 2. Refactor record_model_test_properties()

**Before:** 170 lines with nested conditionals

**After:** ~60 lines using resolver

```python
def record_model_test_properties(...):
    # Resolve bringup status using clear hierarchy
    context = StatusResolutionContext(
        model_test_status=test_metadata.status,
        config_bringup_status=test_metadata.bringup_status,
        test_passed=test_passed,
        comparison_results=comparison_results,
        comparison_config=comparison_config,
    )
    resolution = BringupStatusResolver.resolve(context)
    bringup_status = resolution.bringup_status

    # Assemble tags
    tags = _assemble_test_tags(
        bringup_status=bringup_status,
        ...
    )

    # Record properties
    record_property("tags", _to_marshal_safe(tags))

    # Apply pytest control flow
    _apply_pytest_markers(test_metadata.status, reason)
```

### 3. Extract Helper Functions

```python
def _assemble_test_tags(...) -> dict:
    """Assemble tags dictionary for test reporting."""
    ...

def _apply_pytest_markers(status: ModelTestStatus, reason: str) -> None:
    """Apply pytest skip/xfail markers based on test status."""
    if status == ModelTestStatus.NOT_SUPPORTED_SKIP:
        pytest.skip(reason)
    elif status == ModelTestStatus.KNOWN_FAILURE_XFAIL:
        pytest.xfail(reason)
```

## Resolution Hierarchy

```
1. NOT_SUPPORTED_SKIP → use config_bringup_status (didn't run)
2. NOT_STARTED → placeholder test
3. Has comparison_results:
   ├─ INCORRECT_RESULT → if PCC/ATOL checks fail
   └─ PASSED → if all checks pass
4. No comparison_results:
   ├─ parse_last_bringup_stage() → from file
   └─ UNKNOWN → fallback
5. Apply pytest markers (skip/xfail) separately
```

## Benefits

✅ **Clarity:** Resolution rules explicit and ordered
✅ **Testability:** Pure function, easy to unit test
✅ **Maintainability:** Adding new rules straightforward
✅ **Reduced complexity:** 5 clear rules instead of 7+ nested conditions
✅ **Source tracking:** Know which rule made the decision
✅ **Separation:** Status logic separate from property assembly

## Files

1. **NEW:** `tests/runner/status_resolver.py` - Core resolution logic
2. **NEW:** `tests/runner/test_status_resolver.py` - Unit tests
3. **MODIFY:** `tests/runner/test_utils.py` - Refactor record_model_test_properties()

## Testing Strategy

```python
def test_resolve_not_supported_skip():
    context = StatusResolutionContext(
        model_test_status=ModelTestStatus.NOT_SUPPORTED_SKIP,
        config_bringup_status=BringupStatus.FAILED_RUNTIME,
        test_passed=False,
        comparison_results=None,
        comparison_config=None,
    )
    resolution = BringupStatusResolver.resolve(context)
    assert resolution.bringup_status == BringupStatus.FAILED_RUNTIME
    assert resolution.source == "config_not_supported"

def test_resolve_passed_with_comparison():
    ...

def test_resolve_incorrect_result():
    ...
```

## Documentation

Add clear documentation of the status resolution hierarchy and the semantic difference between:
- **ModelTestStatus (status):** Expected behavior (config-time classification)
- **BringupStatus (bringup_status):** Actual execution outcome (runtime result)
