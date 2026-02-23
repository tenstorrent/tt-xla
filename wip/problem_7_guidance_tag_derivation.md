# Problem 7: Guidance Tag Derivation Separate from Status

**Priority:** MEDIUM

## Problem Description

`_derive_guidance_from_pcc()` and `_derive_guidance_from_status()` are called after status determination, but they examine the same data that was already used for status determination. This leads to redundant data processing.

## Current Code Location

**File:** `tests/runner/test_utils.py`
**Lines:** 330-506 (guidance functions), 664 (combined into single tag)

## Pattern

```python
# Lines 363-450: Process comparison results to determine status
result = _process_comparison_results(comparison_results)
if result == INCORRECT_RESULT:
    status = INCORRECT_RESULT

# Lines 452-506: Re-examine same comparison results for guidance
guidance_pcc = _derive_guidance_from_pcc(comparison_results, ...)
guidance_status = _derive_guidance_from_status(status, ...)

# Line 664: Combine guidances
guidance = guidance_pcc or guidance_status or ""
```

## Issues

1. **Redundant processing:** Same comparison_results examined twice
2. **Scattered logic:** Status and guidance derived in separate places
3. **Hard to understand:** Multiple guidance tags (RAISE_PCC, RAISE_PCC_099, IMPROVE, etc.) without clear docs
4. **Coupling:** Guidance depends on status but processed separately

## Questions to Answer

1. Can status determination and guidance derivation be combined into one pass?
2. Should guidance be part of status object/enum?
3. Are guidance tags actually used? By what?
4. Can we simplify the guidance categories?

## Related Code

- `_derive_guidance_from_pcc()` (lines 452-487)
- `_derive_guidance_from_status()` (lines 490-506)
- `_process_comparison_results()` (lines 363-450)
- Status determination logic in record_model_test_properties

## NOTES

- Should be check why is this added
- If not necessary should be removed in conjunction with `wip/problem_3_complex_status_determination.md`
