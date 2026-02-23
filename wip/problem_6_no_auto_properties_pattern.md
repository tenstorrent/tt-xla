# Problem 6: Confusing no_auto_properties Pattern

**Priority:** MEDIUM

## Problem Description

The `@pytest.mark.no_auto_properties` decorator is used to disable auto-injection at collection time, then properties are manually recorded in the finally block. This two-level system is confusing and requires coordination between conftest.py and test_models.py.

## Current Code Locations

**Collection time:** `tests/runner/conftest.py` lines 69-180
**Runtime:** `tests/runner/test_models.py` lines 242-243, 178-239

## Pattern

```python
# In test_models.py:
@pytest.mark.model_test
@pytest.mark.no_auto_properties  # <-- Disables auto-injection
def test_all_models_torch(...):
    try:
        # ... test logic ...
    finally:
        record_model_test_properties(...)  # <-- Manual recording

# In conftest.py:
def pytest_generate_tests(metafunc):
    if metafunc.definition.get_closest_marker("no_auto_properties"):
        return  # Skip auto-injection
    # ... auto-injection logic ...
```

## Issues

1. **Two-level complexity:** Properties can be injected at collection OR runtime
2. **Requires coordination:** Test must use decorator AND call recording function
3. **Easy to forget:** Could add decorator but forget manual recording
4. **Unclear purpose:** Why have two injection mechanisms?

## Questions to Answer

1. Why do we need both auto-injection and manual recording?
2. Can we consolidate to a single property recording mechanism?
3. What properties are injected at collection time vs runtime?
4. Is auto-injection used by other tests (non-model tests)?

## Related Code

- `pytest_generate_tests()` in conftest.py (lines 69-180)
- Property recording in test_utils.py
- Other tests that might use auto-injection

## NOTES

- Would love more detailed explanation why is no_auto_properties needed