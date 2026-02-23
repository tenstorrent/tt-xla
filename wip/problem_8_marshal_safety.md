# Problem 8: Marshal Safety Conversion Inconsistent

**Priority:** LOW

## Problem Description

The `_to_marshal_safe()` function exists due to pytest-forked incompatibility with non-builtin types, but it's not used consistently. There's no compile-time guarantee that all values passed to `record_property()` are marshal-safe.

## Current Code Location

**File:** `tests/runner/test_utils.py`
**Function:** `_to_marshal_safe()` (lines 295-327)
**Usage:** Scattered throughout record_model_test_properties (lines 509-682)

## Inconsistent Usage

```python
# Line 669 - wrapped correctly
record_property("error_message", _to_marshal_safe(reason))

# Line 672 - wrapped correctly
record_property("tags", _to_marshal_safe(tags))

# Line 673 - NOT wrapped (happens to be string)
record_property("owner", "tt-xla")

# Line 675 - manual str() instead of wrapper
record_property("group", str(model_info.group))
```

## Issues

1. **Inconsistent wrapping:** Some values wrapped, some not
2. **No compile-time guarantee:** Easy to forget wrapping
3. **Manual conversions:** Some code uses str() directly instead of wrapper
4. **Risk of crashes:** Passing non-builtin types causes pytest-forked to fail

## Questions to Answer

1. Can we create a type-safe wrapper for record_property()?
2. Should we enforce wrapping at compile time (type hints)?
3. Can we auto-wrap all values in a custom record_property wrapper?
4. Are there values that should NOT be wrapped?

## Related Code

- `_to_marshal_safe()` implementation (lines 295-327)
- All `record_property()` calls in test_utils.py
- pytest-forked documentation on marshal safety

## NOTES

- This should probably be different PR but would love to see plan for this
