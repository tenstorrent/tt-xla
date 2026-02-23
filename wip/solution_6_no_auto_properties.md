# Solution 6: Understanding no_auto_properties Pattern

## Explanation: Why This Two-Level System Exists

The `@pytest.mark.no_auto_properties` pattern exists because of **pytest's two-phase architecture** (collection vs execution) combined with model tests needing **runtime-dependent properties**.

## What Auto-Injection Does

Located in `tests/conftest.py` (lines 69-180), auto-injection runs during pytest's **collection phase** (before tests execute):

1. Extracts `@pytest.mark.record_test_properties(...)` marker
2. Builds a `tags` dictionary with test metadata
3. Attaches to `item.user_properties` immediately
4. Properties available in report without test execution

**Example (non-model test):**
```python
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_info=MODEL_INFO,
    bringup_status=BringupStatus.PASSED,  # Static, known upfront
)
def test_resnet_inference(trace_tester):
    trace_tester.test()
```

## Why Model Tests Can't Use Auto-Injection

Model tests have **runtime-dependent properties** that aren't known at collection time:

### Runtime Properties (can't be known upfront):
1. **Comparison results** - PCC scores, ATOL values (from test execution)
2. **Bringup status** - PASSED vs INCORRECT_RESULT vs failure stage
3. **Execution failures** - Exception info captured during test
4. **Failing reasons** - Dynamic categorization based on test outcome
5. **Model size metrics** - Computed during test execution

### Static Properties (could be auto-injected):
- Test name, model info, run mode, parallelism
- But since we need manual recording anyway, we disable auto-injection entirely

## Why Both Mechanisms Exist

**Technical reason:** pytest's `record_property` fixture only works at **runtime** (inside test function), but properties recorded at collection time via `item.user_properties` get included in reports without needing the fixture.

**Trade-off:**
- **Collection-time (auto-injection)**: Fast, simple, but only static data
- **Runtime (manual recording)**: Dynamic, captures test results, but requires coordination

## The Coordination Pattern

```python
@pytest.mark.model_test
@pytest.mark.no_auto_properties  # Disable auto-injection
def test_all_models_torch(...):
    try:
        # Test execution
        comparison_result = tester.test()
    finally:
        # Manual recording captures runtime results
        record_model_test_properties(
            comparison_results=comparison_result,
            test_passed=succeeded,
            # ... runtime data
        )
```

## Can This Be Simplified?

**Current complexity:** Tests must use `@pytest.mark.no_auto_properties` AND call `record_model_test_properties()` in finally blocks.

**Simplification with Problem 1 solution:**
Once property recording moves to pytest hooks (Problem 1), the pattern becomes cleaner:
1. Store metadata on `request.node` before test runs
2. Hook automatically records properties (including runtime results)
3. No need for manual finally block recording
4. `@pytest.mark.no_auto_properties` still needed to avoid double recording

## Conclusion

**Is the two-level system necessary?** Yes, given pytest's architecture:
- Collection-time can't access runtime results
- Runtime recording can't affect test selection (depends on markers at collection)

**Can coordination be improved?** Yes, with Problem 1 hook-based solution:
- Eliminates manual `record_model_test_properties()` call
- Still need `@pytest.mark.no_auto_properties` to disable auto-injection
- But the pattern becomes: "disable auto-injection, hook handles runtime recording"

## Recommendation

Keep the two-level system but simplify coordination:
1. Implement Problem 1 (hook-based recording)
2. Remove manual recording from finally blocks
3. Document that `@pytest.mark.no_auto_properties` triggers automatic hook recording
4. Pattern becomes: "Use marker to disable auto-injection; hook records everything"
