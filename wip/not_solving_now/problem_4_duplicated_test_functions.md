# Problem 4: Duplicated Torch/JAX Test Functions

**Priority:** MEDIUM

## Problem Description

`test_all_models_torch` and `test_all_models_jax` are nearly identical functions with the same decorators and structure, differing only in the framework parameter passed to `_run_model_test_impl`.

## Current Code Location

**File:** `tests/runner/test_models.py`
**Lines:** 243-353

## Duplication Pattern

```python
@pytest.mark.model_test
@pytest.mark.no_auto_properties
@pytest.mark.parametrize("run_mode", [...])
@pytest.mark.parametrize("parallelism", [...])
@pytest.mark.parametrize("test_entry", [...])
def test_all_models_torch(...):
    _run_model_test_impl(..., framework=Framework.TORCH, ...)

# Nearly identical:
@pytest.mark.model_test
@pytest.mark.no_auto_properties
@pytest.mark.parametrize("run_mode", [...])
@pytest.mark.parametrize("parallelism", [...])
@pytest.mark.parametrize("test_entry", [...])
def test_all_models_jax(...):
    _run_model_test_impl(..., framework=Framework.JAX, ...)
```

## Issues

1. **Code duplication:** Same decorators and structure repeated
2. **Maintenance burden:** Changes need to be made in both places
3. **Inconsistency risk:** Easy to update one but forget the other
4. **Violates DRY principle**

## Questions to Answer

1. Can we unify these into a single parametrized test with framework as a parameter?
2. Are there any framework-specific decorators/filters that prevent unification?
3. Would pytest collection be affected by unification?
4. Do test names/IDs need to maintain current format for compatibility?

## Related Code

- `_run_model_test_impl()` - called by both functions
- Framework enum definition
- Pytest parametrization in conftest.py


## NOTES

- This is not nessary, it is easier for picking correct tests to run. So abort on this problem, but save it in subfolder for the problems that we will not solve right now