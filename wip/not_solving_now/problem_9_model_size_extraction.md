# Problem 9: Redundant Model Size Extraction

**Priority:** LOW

## Problem Description

Model size is extracted separately in the finally block based on framework and tester existence, but it's only used for benchmark results. This logic could be part of the tester interface or benchmark result generation flow.

## Current Code Location

**File:** `tests/runner/test_models.py`
**Lines:** 180-182

```python
comparison_config = tester._comparison_config if tester else None
model_size = None
if framework == Framework.TORCH and tester is not None:
    model_size = getattr(tester, "_model_size", None)
```

## Issues

1. **Framework-specific:** Only extracted for Torch
2. **Attribute access pattern:** Uses getattr with None default (suggests optional)
3. **Separated from usage:** Extracted here but used in benchmark results (line 627)
4. **Redundant checks:** tester existence checked after comparison_config already checked it

## Questions to Answer

1. Should model_size be a standard tester interface property?
2. Can benchmark result generation extract this itself?
3. Why is this only for Torch? Do JAX models not have size?
4. Should this be part of comparison_config?

## Related Code

- Tester classes and their interfaces
- Benchmark result generation (line 627 in test_models.py)
- `_model_size` attribute in tester implementations

## NOTES

- This is not necessary. So abort on this problem, but save it in subfolder for the problems that we will not solve right now