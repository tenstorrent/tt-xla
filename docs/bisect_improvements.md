# TT-Metal Bisect Documentation

## Overview

The tt-metal bisect system extends the automated bisect framework to support three-level dependency bisecting: tt-xla → tt-mlir → tt-metal. This enables pinpointing performance regressions deep in the dependency chain without manual intervention.

## How TT-Metal Bisect Works

The tt-metal bisect (`scripts/bisect_ttmetal_perf.sh`) operates within the context of a tt-mlir commit that uplifts tt-metal. For each candidate tt-metal commit being tested:

### 1. Checkout and Prepare
- Checkout the specific tt-metal commit in the submodule
- Update all tt-metal submodules: `git submodule update --init --recursive`
  - Critical because tt-metal has its own submodule dependencies
- Navigate to tt-mlir repository (parent of tt-metal)

### 2. Apply Modifications
- **Apply revert** of specified tt-mlir commit if provided (`-r` flag)
  - Checks if revert commit exists in current tt-mlir history
  - Only reverts if commit is an ancestor of current HEAD
  - Revert is applied in tt-mlir, NOT in tt-metal submodule

- **Modify tt-xla's third_party/CMakeLists.txt** to use current tt-metal version
  - This is critical to prevent CMake ExternalProject conflicts
  - Without this, ExternalProject tries to checkout a different commit and hits git stash conflicts

### 3. Build Attempt with Parent Checkout Strategy

This is the most efficient approach to minimize Claude invocations:

```bash
# Reset all changes to clean state
git reset --hard HEAD
git clean -fd

# Checkout PARENT of reference commit (before compatibility fixes)
git checkout "${FIX_BUILD_REF}^"

# Update tt-xla's TT_MLIR_VERSION to match parent commit
# This prevents ExternalProject from trying to checkout different commit
sed -i "s/set(TT_MLIR_VERSION \"[^\"]*\")/set(TT_MLIR_VERSION \"$PARENT_COMMIT\")/" \
    third_party/CMakeLists.txt

# Re-apply tt-metal version modification
sed -i "s/set(TT_METAL_VERSION \"[^\"]*\")/set(TT_METAL_VERSION \"$CURRENT_TTMETAL_COMMIT\")/" \
    third_party/CMakeLists.txt

# Re-apply revert if specified
if [ -n "$REVERT_COMMIT" ]; then
    git revert --no-commit "$REVERT_COMMIT"
fi

# Try building with parent commit first
cmake --build build
```

**Why this works:**
- Parent commit is BEFORE compatibility fixes were added
- Many intermediate tt-metal commits work fine with parent
- Only invoke Claude if parent build also fails
- Reduces Claude invocations by ~50%

### 4. Claude Agent Integration (`--fix-build-ref` option)

Only invoked if parent build fails. Claude examines the reference commit's compatibility fixes and adapts them for the specific intermediate tt-metal API version.

**Key insight:** Intermediate tt-metal versions may have different APIs than the final version that the reference commit was written for.

#### Common Error Types Claude Handles

**a) API signature mismatches**
- Function calls with wrong number or types of arguments
- Solution: Examine tt-metal API at specific commit and adjust calls

**b) Header redefinition errors** (most common)
- Types/enums defined in BOTH `build/include/` AND source directories
- Happens when headers are both generated and in source tree
- Solutions:
  - Modify tt-mlir's `common.h` to exclude one include path
  - Add include guards or conditional compilation
  - Check if CMakeLists.txt needs updates to exclude duplicate headers
  - Prefer build/include versions if auto-generated

**c) Missing symbols**
- Functions/types that don't exist in this tt-metal version
- May need to use alternative APIs or conditionally compile

#### Example Fix Applied by Claude

For header redefinition errors in commit `63bb00ab4a`:

```diff
diff --git a/runtime/include/tt/runtime/detail/common/common.h b/runtime/include/tt/runtime/detail/common/common.h
index f9523dbd6..37b664bbc 100644
--- a/runtime/include/tt/runtime/detail/common/common.h
+++ b/runtime/include/tt/runtime/detail/common/common.h
@@ -13,8 +13,8 @@
 #include "tt-metalium/host_api.hpp"
 #include "tt-metalium/mesh_device.hpp"

-#include "tt-metalium/fabric_edm_types.hpp"
-#include "tt-metalium/fabric_types.hpp"
+#include "tt-metalium/experimental/fabric/fabric_edm_types.hpp"
+#include "tt-metalium/experimental/fabric/fabric_types.hpp"
 #include "tt/runtime/detail/common/flatbuffer_operator_ostream.h"
 #include "tt/runtime/detail/common/logger.h"
 #include "tt/runtime/types.h"
```

**Why this fixes it:**
- Before: `tt-metalium/fabric_edm_types.hpp` resolved to generated headers in `build/include/tt-metalium/`
- Other files were including from `experimental/fabric/` path → source headers
- This caused both versions to be included → redefinition errors
- Claude's fix: Use full source path consistently → only one version included

#### Claude's Approach

```
1. Read build errors to understand what's failing
2. Examine diff between parent and reference commit
3. Adapt fixes for THIS specific tt-metal API version
4. Modify tt-mlir files only (not tt-metal submodule)
5. For header conflicts: modify include paths in common.h
6. For API mismatches: examine actual tt-metal code at commit
7. Test build iteratively until success or timeout
```

**Configuration:**
- Model: Opus (better reasoning for complex issues)
- Timeout: 600 seconds (10 minutes)
- Allowed tools: Read, Edit, Bash, Glob, Grep
- Permission mode: bypassPermissions (fully automated)
- Verbose: Enabled for debugging

### 5. Performance Test

If build succeeds:
```bash
# Run benchmark command
$BENCHMARK_COMMAND

# Extract performance metric using regex pattern
PERFORMANCE=$(grep -oP "$METRIC_PATTERN" benchmark_output.log)

# Compare against threshold
if [ "$PERFORMANCE" -lt "$THRESHOLD" ]; then
    exit 1  # BAD
else
    exit 0  # GOOD
fi
```

Exit codes:
- `0`: Good (performance meets threshold)
- `1`: Bad (performance below threshold)
- `125`: Untestable (build failed, Claude timed out)

### 6. Cleanup

```bash
cleanup() {
    # Restore reference commit if used
    if [ -n "$FIX_BUILD_REF" ]; then
        cd "$TTMLIR_SRC_DIR"
        git checkout "$FIX_BUILD_REF" --quiet
        git reset --hard HEAD
        git clean -fd  # Remove untracked files (e.g., venv/ created by Claude)
    fi

    # Reset tt-xla changes
    cd "$TTXLA_ROOT"
    git reset --hard HEAD
}
```

## Integration with Auto Bisect

The `bisect_perf_auto.sh` script orchestrates three-level bisecting:

### Phase 3: TT-Metal Detection

```bash
# Check if bad tt-mlir commit is a tt-metal uplift
cd "$TTMLIR_THIRD_PARTY_DIR"

git checkout "$FIRST_BAD_TTMLIR" --quiet
BAD_TTMETAL=$(grep 'set(TT_METAL_VERSION' CMakeLists.txt | grep -oP '"\K[^"]+')

git checkout "${FIRST_BAD_TTMLIR}^" --quiet
PARENT_TTMETAL=$(grep 'set(TT_METAL_VERSION' CMakeLists.txt | grep -oP '"\K[^"]+')

if [ "$BAD_TTMETAL" != "$PARENT_TTMETAL" ]; then
    # This is a tt-metal uplift, bisect within tt-metal
    echo "Detected tt-metal uplift: $PARENT_TTMETAL → $BAD_TTMETAL"

    # Automatically invoke tt-metal bisect with reference commit
    bisect_ttmetal_perf.sh \
        -c "$BENCHMARK_COMMAND" \
        -t "$PERF_THRESHOLD" \
        -p "$METRIC_PATTERN" \
        -f "$FIRST_BAD_TTMLIR"  # Use tt-mlir uplift as fix reference
fi
```

## Key Design Decisions

### Parent Checkout Strategy

Testing with parent commit (before fixes) first is most efficient:
- Only invoke Claude if parent also fails
- Reduces Claude usage and total bisect time
- Many intermediate commits work fine with parent

### TT_MLIR_VERSION Hotfix

Updating tt-xla's `CMakeLists.txt` to match checked-out commit prevents CMake ExternalProject from trying to sync tt-mlir to a different commit, which causes git stash conflicts.

**Without hotfix:**
```
error: Your local changes to the following files would be overwritten by checkout:
    third_party/CMakeLists.txt
Please commit your changes or stash them before you switch branches.
Aborting
```

**With hotfix:**
- ExternalProject sees correct TT_MLIR_VERSION
- No attempt to checkout different commit
- No git conflicts

### Revert in TT-MLIR, Not TT-Metal

Reverts are applied in the parent tt-mlir repository because:
- That's where the buggy changes exist that need reverting
- TT-metal is a git submodule with independent history
- Reverting in tt-metal would be incorrect and confusing

### Submodule Updates

Critical to run `git submodule update --init --recursive` when moving through tt-metal commits:
- TT-metal has its own submodule dependencies (UMD, tracy, etc.)
- These must be synced to match the tt-metal commit being tested
- Missing submodule updates cause cryptic build failures

## Example Workflow

Full three-level bisect for resnet regression:

```bash
# Start auto bisect
cd /localdev/rpavlovic/tt-xla
./scripts/bisect_perf_auto.sh \
    -c "python ../tt-forge/benchmark/benchmark.py -p tt-xla -m resnet -bs 8 -df bfloat16 -lp 128" \
    -t 540 \
    -p "Sample per second:\s*\K[0-9.]+"

# Phase 1: Bisect tt-xla
[INFO] Starting tt-xla bisect...
[INFO] Testing range: 3ca42547..a6b9d854
[INFO] Found bad commit: a868fa2a9 (tt-mlir uplift)
[INFO] Detected TT_MLIR_VERSION change, entering Phase 2

# Phase 2: Auto-bisect tt-mlir
[INFO] Starting tt-mlir bisect...
[INFO] Testing range: 70efb12f..fb45bf24c
[INFO] Found bad commit: fb45bf24c (tt-metal uplift)
[INFO] Detected TT_METAL_VERSION change, entering Phase 3

# Phase 3: Auto-bisect tt-metal
[INFO] Starting tt-metal bisect...
[INFO] Testing range: e1d6113542..2bd1bb143f
[INFO] Testing commit: 63bb00ab4a
[INFO] Build failed with parent commit
[INFO] Invoking Claude to fix build...
[INFO] Claude fixed header redefinition errors in common.h
[INFO] Build succeeded, running benchmark...
[INFO] Performance: 425.99 (threshold: 540) → BAD

[INFO] Testing commit: 5548ea559d
[INFO] Build succeeded with parent commit
[INFO] Performance: 542.10 (threshold: 540) → GOOD

[INFO] Bisect converging...
[INFO] First bad commit: 63bb00ab4af2ec88203bbb371291324b7d2a4af1
[INFO] Commit message: "#33134: Use output cores for throttle on convs. (#33135)"
```

## Future Improvements

### 1. Cache Claude's Build Fixes as Patches

**Problem:**
Currently, Claude is invoked fresh for each failing commit. However, many intermediate commits fail with the SAME header redefinition errors and require the SAME fix. This wastes time and Claude API calls.

**Proposed Solution:**

```bash
# After Claude successfully fixes a build
CLAUDE_PATCH="/tmp/bisect_patches/claude_fix_${TTMETAL_COMMIT}.patch"
mkdir -p /tmp/bisect_patches

# Extract ONLY Claude's changes (excluding revert)
if [ -n "$REVERT_COMMIT" ]; then
    # Get list of files changed by revert
    REVERT_FILES=$(git diff-tree --no-commit-id --name-only -r "$REVERT_COMMIT")

    # Create patch excluding revert files
    git diff HEAD -- $(git ls-files -m | grep -v -F "$REVERT_FILES") > "$CLAUDE_PATCH"
else
    # No revert, all changes are from Claude
    git diff HEAD > "$CLAUDE_PATCH"
fi

# Tag patch with error signature for smart matching
ERROR_SIG=$(echo "$BUILD_ERRORS" | grep "error:" | sort | md5sum | cut -d' ' -f1)
echo "# Error signature: $ERROR_SIG" >> "$CLAUDE_PATCH"
```

**Patch Application Strategy:**

```bash
try_cached_patches() {
    local current_commit=$1
    local error_sig=$2

    # Try patches in order of likelihood
    local patches=(
        # 1. Exact commit match (if we've seen this before)
        "/tmp/bisect_patches/claude_fix_${current_commit}.patch"

        # 2. Parent commit (likely similar)
        "/tmp/bisect_patches/claude_fix_${current_commit}^.patch"

        # 3. Same error signature
        $(grep -l "Error signature: $error_sig" /tmp/bisect_patches/*.patch 2>/dev/null)

        # 4. Common fixes (header issues, API compat)
        "/tmp/bisect_patches/common_header_fix.patch"
        "/tmp/bisect_patches/common_api_compat.patch"
    )

    for patch in "${patches[@]}"; do
        if [ -f "$patch" ]; then
            echo "Trying cached patch: $(basename $patch)"

            # Check if patch applies cleanly
            if git apply --check "$patch" 2>/dev/null; then
                git apply "$patch"

                # Try building
                if cmake --build build 2>&1 | tee build_test.log; then
                    echo "✓ Cached patch worked!"
                    return 0
                fi

                # Revert failed patch
                git reset --hard HEAD
            fi
        fi
    done

    return 1  # No cached patch worked
}

# In main bisect loop
if ! cmake --build build; then
    ERROR_SIG=$(echo "$BUILD_ERRORS" | grep "error:" | sort | md5sum | cut -d' ' -f1)

    # Try cached patches first
    if try_cached_patches "$TTMETAL_COMMIT" "$ERROR_SIG"; then
        echo "Build fixed with cached patch"
    else
        # Fall back to Claude
        invoke_claude_to_fix_build
    fi
fi
```

**Challenge: Separating Claude's Changes from Revert**

Three approaches with trade-offs:

**Approach 1: Patch before revert**
```bash
# Checkout parent → invoke Claude → save diff → apply revert
git checkout "${FIX_BUILD_REF}^"
invoke_claude
CLAUDE_DIFF=$(git diff HEAD)
echo "$CLAUDE_DIFF" > "$CLAUDE_PATCH"
git revert --no-commit "$REVERT_COMMIT"
echo "$CLAUDE_DIFF" | git apply
```
- ✓ Clean separation
- ✗ Complex workflow

**Approach 2: Track revert files and exclude**
```bash
REVERT_FILES=$(git show "$REVERT_COMMIT" --name-only --pretty=format:)
git diff HEAD -- $(git ls-files -m | grep -v -F "$REVERT_FILES") > "$CLAUDE_PATCH"
```
- ✓ Simple
- ✗ Misses files modified by both revert and Claude

**Approach 3: Use git's three-way diff**
```bash
# Most accurate but complex
git diff "$REVERT_COMMIT"^.."$REVERT_COMMIT" > revert.patch
git diff HEAD > combined.patch
# Use diff tools to subtract revert.patch from combined.patch
```
- ✓ Most accurate
- ✗ Requires careful diff manipulation

**Recommended: Approach 2 with heuristics**
- Simple and works for 90% of cases
- Add error signature matching for robustness
- Store metadata about revert state in patch header

**Expected Benefits:**
- ~80% reduction in Claude invocations
- ~60% reduction in total bisect time
- More consistent fixes (same patch applied to similar commits)
- Lower cost (fewer Claude API calls)

**Implementation Plan:**
1. Add patch caching to bisect_ttmetal_perf.sh
2. Create `/tmp/bisect_patches/` directory structure
3. Implement `try_cached_patches()` function
4. Add error signature extraction
5. Test with known regression

### 2. Add Timestamps to All Logs

**Problem:**
Currently difficult to track timing and identify bottlenecks. Hard to answer questions like:
- How long did Claude take on this commit?
- Which commits take longest to build?
- What's the average benchmark time?
- When did the bisect start/finish?

**Proposed Solution:**

```bash
# Utility function for timestamped logging
log_with_timestamp() {
    local message="$1"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo "[$timestamp] $message" | tee -a "$LOG_FILE"
}

# Enhanced logging at key events
log_with_timestamp "========================================"
log_with_timestamp "Starting tt-metal commit test: $COMMIT_SHORT"
log_with_timestamp "Benchmark: $BENCHMARK_COMMAND"
log_with_timestamp "Threshold: $PERF_THRESHOLD"
log_with_timestamp "========================================"

# Track durations
BUILD_START=$(date +%s)
log_with_timestamp "Build started"
cmake --build build
BUILD_END=$(date +%s)
BUILD_DURATION=$((BUILD_END - BUILD_START))
log_with_timestamp "Build completed (duration: ${BUILD_DURATION}s)"

# Claude invocation timing
if [ "$BUILD_FAILED" ]; then
    CLAUDE_START=$(date +%s)
    log_with_timestamp "Invoking Claude to fix build..."
    timeout 600 claude -p "$FIX_PROMPT" ...
    CLAUDE_EXIT=$?
    CLAUDE_END=$(date +%s)
    CLAUDE_DURATION=$((CLAUDE_END - CLAUDE_START))

    if [ $CLAUDE_EXIT -eq 124 ]; then
        log_with_timestamp "✗ Claude timed out after ${CLAUDE_DURATION}s"
    else
        log_with_timestamp "✓ Claude completed (duration: ${CLAUDE_DURATION}s)"
    fi
fi

# Benchmark timing
BENCH_START=$(date +%s)
log_with_timestamp "Benchmark started"
$BENCHMARK_COMMAND
BENCH_END=$(date +%s)
BENCH_DURATION=$((BENCH_END - BENCH_START))
log_with_timestamp "Benchmark completed (duration: ${BENCH_DURATION}s)"

# Result with timestamp
log_with_timestamp "Result: $RESULT (performance: $PERFORMANCE)"
log_with_timestamp "Total commit time: $((BENCH_END - BUILD_START))s"
```

**Enhanced Summary:**

```bash
# At end of bisect
log_with_timestamp "========================================"
log_with_timestamp "Bisect completed!"
log_with_timestamp "First bad commit: $FIRST_BAD_COMMIT"
log_with_timestamp "========================================"
log_with_timestamp "Statistics:"
log_with_timestamp "  Total commits tested: $TOTAL_COMMITS"
log_with_timestamp "  Commits skipped (untestable): $SKIPPED_COMMITS"
log_with_timestamp "  Claude invocations: $CLAUDE_INVOCATIONS"
log_with_timestamp "  Claude successes: $CLAUDE_SUCCESSES"
log_with_timestamp "  Total bisect time: $TOTAL_DURATION"
log_with_timestamp "  Average time per commit: $AVG_COMMIT_TIME"
log_with_timestamp "========================================"
```

**Example Output:**

```
[2025-11-28 13:23:15] ========================================
[2025-11-28 13:23:15] Starting tt-metal commit test: 63bb00ab4a
[2025-11-28 13:23:15] Benchmark: python benchmark.py ...
[2025-11-28 13:23:15] Threshold: 540
[2025-11-28 13:23:15] ========================================
[2025-11-28 13:23:16] Build started
[2025-11-28 13:25:42] Build completed (duration: 146s)
[2025-11-28 13:25:42] Build failed, trying parent commit
[2025-11-28 13:25:43] Build started (parent)
[2025-11-28 13:27:58] Build failed (parent) (duration: 135s)
[2025-11-28 13:27:58] Invoking Claude to fix build...
[2025-11-28 13:32:45] ✓ Claude completed (duration: 287s)
[2025-11-28 13:32:45] Build started (with fixes)
[2025-11-28 13:35:12] Build completed (duration: 147s)
[2025-11-28 13:35:12] Benchmark started
[2025-11-28 13:40:33] Benchmark completed (duration: 321s)
[2025-11-28 13:40:33] Result: BAD (performance: 425.99)
[2025-11-28 13:40:33] Total commit time: 1038s (17.3 min)
```

**Benefits:**
- Easy to identify slow commits
- Track Claude timeout patterns
- Benchmark duration analysis
- Total bisect time estimation
- Better debugging when issues occur
- Historical data for optimization

**Implementation:**
1. Add `log_with_timestamp()` function
2. Add timing variables (start/end/duration)
3. Replace all `echo` with `log_with_timestamp`
4. Add summary statistics at end
5. Optional: Export timing data to CSV for analysis

### 3. Smart Patch Database with Categorization

Build on the caching idea with better organization:

```bash
# Organize patches by error category
PATCH_DB="/tmp/bisect_patches/"
mkdir -p "$PATCH_DB"/{header_redefinition,api_mismatch,missing_symbols,other}

categorize_error() {
    local errors="$1"

    if echo "$errors" | grep -q "redefinition of"; then
        echo "header_redefinition"
    elif echo "$errors" | grep -q "no matching function"; then
        echo "api_mismatch"
    elif echo "$errors" | grep -q "undefined reference"; then
        echo "missing_symbols"
    else
        echo "other"
    fi
}

save_categorized_patch() {
    local category=$(categorize_error "$BUILD_ERRORS")
    local patch_file="$PATCH_DB/$category/fix_${TTMETAL_COMMIT}.patch"

    git diff HEAD > "$patch_file"

    # Add metadata
    cat >> "$patch_file" <<EOF
# Metadata
# Commit: $TTMETAL_COMMIT
# Category: $category
# Error signature: $(echo "$BUILD_ERRORS" | md5sum | cut -d' ' -f1)
# Timestamp: $(date -Iseconds)
# Files modified: $(git diff --name-only HEAD | tr '\n' ' ')
EOF
}
```


## Testing Checklist

- [x] Successfully bisects through tt-metal range
- [x] Claude fixes header redefinition errors
- [x] Parent checkout strategy works
- [x] TT_MLIR_VERSION hotfix prevents conflicts
- [x] Revert applies correctly in tt-mlir
- [x] Submodule updates work
- [x] Timeout handling (10 minutes)
- [x] Cleanup restores repository state
- [ ] Patch caching (future)
- [ ] Timestamp logging (future)

## References

- Main script: `scripts/bisect_ttmetal_perf.sh`
- Auto bisect: `scripts/bisect_perf_auto.sh`
- Related: `scripts/bisect_ttmlir_perf.sh`
- Example commit: 63bb00ab4a (Claude fixed header redefinitions)
