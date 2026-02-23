# Solution 7: Guidance Tag Derivation

## What Are Guidance Tags?

Guidance tags are **automation hints** for test configuration improvements. They suggest actions to test maintainers for threshold tuning and test configuration updates.

## Guidance Tag Types

### PCC-Based Guidance
- **ENABLE_PCC**: PCC check disabled but measured PCC safely above threshold → suggest enabling
- **ENABLE_PCC_099**: Same but for thresholds ≥ 0.99
- **RAISE_PCC**: Measured PCC exceeds next threshold step → suggest raising
- **RAISE_PCC_099**: Measured PCC exceeds 0.99 → suggest raising to 0.99

### Status-Based Guidance
- **RM_XFAIL**: Test marked KNOWN_FAILURE_XFAIL but passing → suggest removing xfail
- **ADD_CONFIG**: Test marked UNSPECIFIED but ready → suggest adding proper config

## Current Consumers

1. **`.github/scripts/summarize_junit_xmls.py`** (primary)
   - Normalizes guidance field for report generation
   - Included in CSV/tabular output

2. **Dashboard/Reporting** (implicit)
   - Guidance included in JUnit XML properties
   - Likely consumed by Superset dashboards

3. **Future Automation** (intended but not implemented)
   - Designed for CI automation to promote/demote tests
   - No active automation currently exists

## Are Guidance Tags Necessary?

**Verdict: OPTIONAL - Not critical but useful**

### Arguments for Keeping:
✅ Zero computational overhead (O(1) per test)
✅ Useful hints for test maintainers
✅ Dashboard visibility for test improvements
✅ Infrastructure ready for future automation
✅ Can be removed without breaking tests

### Arguments for Removing:
❌ Not actively used beyond JUnit reporting
❌ Re-examines data already processed for status
❌ Lacks comprehensive documentation
❌ No test coverage for guidance functions

## Relationship with Problem 3

**Current inefficiency:** Both status determination and guidance derivation examine the same inputs:
- `comparison_results` processed twice
- `test_metadata.status` examined twice
- PCC metrics checked with different logic

**Example of redundancy:**
- Status checks: `if PCC < required_pcc` → INCORRECT_RESULT
- Guidance checks: `if PCC > required_pcc + buffer` → RAISE_PCC

## Recommendation: KEEP but INTEGRATE

### Integration with Problem 3 Solution

Combine guidance derivation into `BringupStatusResolver`:

```python
class BringupStatusResolver:
    def resolve(self, context):
        """
        Single pass determines status and guidance simultaneously.
        Returns: (bringup_status, reason, guidance_tags)
        """
        # Determine status
        bringup_status = self._resolve_status(context)

        # Derive guidance in same pass
        guidance = self._derive_guidance(
            context.comparison_results,
            context.comparison_config,
            context.model_test_status,
            bringup_status
        )

        return bringup_status, reason, guidance
```

### Benefits of Integration:
✅ Single pass over comparison results
✅ Clear coupling between status and guidance
✅ Easier to test (one resolver)
✅ Reduced code duplication
✅ Maintains useful functionality

## Action Plan

1. **Integrate into BringupStatusResolver** (Problem 3 solution)
   - Move guidance derivation into status resolver
   - Return guidance as part of resolution output
   - Eliminate redundant data processing

2. **Add test coverage**
   - Unit tests for guidance derivation with edge cases
   - Integration tests verifying guidance in reports

3. **Document guidance tags**
   - Add reference to test documentation
   - Explain when each tag appears
   - Provide examples

4. **Optional: Add filter to summarize_junit_xmls.py**
   - `--guidance` filter for automation-ready queries

## Files Involved

- `tests/runner/test_utils.py` (lines 330-360, 452-506) - Guidance functions
- `.github/scripts/summarize_junit_xmls.py` - Consumer
- `tests/runner/status_resolver.py` (NEW from Problem 3) - Integration target

## Summary

| Aspect | Details |
|--------|---------|
| **Purpose** | Automation hints for threshold tuning and config updates |
| **Current Use** | JUnit XML reporting, dashboard visibility |
| **Computational Cost** | Negligible (O(1) per test) |
| **Recommendation** | Keep and integrate into Problem 3 BringupStatusResolver |
| **Benefits** | Useful for maintainers, future-proofed for automation |
| **Risk of Removal** | Low - only informational value |
