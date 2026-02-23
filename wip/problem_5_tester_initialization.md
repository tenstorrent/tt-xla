# Problem 5: Tester Initialization Not Abstracted

**Priority:** MEDIUM

## Problem Description

There are 4 different tester initialization code paths based on framework + parallelism combination, with no abstraction pattern. This makes it hard to add new testers or parallelism strategies.

## Current Code Location

**File:** `tests/runner/test_models.py`
**Lines:** 113-154

## Current Pattern

```python
if framework == Framework.TORCH:
    tester = DynamicTorchModelTester(...)
elif framework == Framework.JAX:
    if parallelism in (TENSOR_PARALLEL, DATA_PARALLEL):
        tester = DynamicJaxMultiChipModelTester(...)
    else:
        if model_info.source == EASYDEL:
            tester = DynamicJaxMultiChipModelTester(...)
        else:
            tester = DynamicJaxModelTester(...)
```

## Issues

1. **No abstraction:** Direct if/elif chains instead of factory pattern
2. **Hard to extend:** Adding new framework or parallelism requires modifying this code
3. **Parameter complexity:** Each tester has different required parameters
4. **Framework-specific logic:** model_info.source check only for JAX

## Questions to Answer

1. Should we use a factory pattern or registry for tester creation?
2. Can tester parameters be standardized across implementations?
3. Should parallelism selection be part of tester interface?
4. Can we move framework-specific logic (like EASYDEL check) into tester classes?

## Related Code

- `DynamicTorchModelTester` class
- `DynamicJaxModelTester` class
- `DynamicJaxMultiChipModelTester` class
- Tester base class or interface (if exists)


## NOTES

- This should probably be separate PR but would love to see a proposal how this can be done
