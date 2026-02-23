# Problem 3: Complex Status Determination Logic

**Priority:** MEDIUM-HIGH

## Problem Description

The `record_model_test_properties()` function has 7+ conditional branches to determine final bringup status, with multiple sources of truth making it hard to follow and maintain.

## Current Code Location

**File:** `tests/runner/test_utils.py`
**Function:** `record_model_test_properties` (lines 509-682)

## Current Decision Tree

```
1. If NOT_SUPPORTED_SKIP → skip() at end
2. Elif NOT_STARTED → just record as is
3. Elif comparison_results present:
   a. If INCORRECT_RESULT → set to INCORRECT_RESULT
   b. Elif test_passed → set to PASSED
4. Else (no results):
   a. Use parse_last_bringup_stage()
   b. Else use UNKNOWN
5. If KNOWN_FAILURE_XFAIL → xfail() at end
```

## Multiple Sources of Truth

- `test_metadata.status` (from config)
- `test_metadata.bringup_status` (from config)
- `comparison_results` (runtime)
- `test_metadata.failing_reason` (can be set by exception handler)
- Parsed bringup stage from file
- `test_passed` parameter

## Issues

1. **Hard to reason about:** Many conditional branches with nested logic
2. **Multiple state sources:** Status can come from 5+ different places
3. **Scattered logic:** Status determination mixed with property recording
4. **Testing difficulty:** Hard to write comprehensive tests for all branches

## Questions to Answer

1. Can we create a clear hierarchy of status sources (e.g., runtime > config > parsed)?
2. Should status determination be a separate function/class?
3. Can we reduce the number of status sources?
4. What's the difference between `status` and `bringup_status`?

## Related Code

- `parse_last_bringup_stage()` in test_utils.py
- `_process_comparison_results()` in test_utils.py (lines 363-450)
- Status markers applied in conftest.py
