# PCC artifacts for `test_matmul_mp.py`

Snapshots of measurements and the tooling used to produce them. Regenerate
when the predicate in `test_matmul_mp.py::_is_known_pcc_failure` or any
upstream component (tt-mlir, tt-metal, evaluator thresholds) changes.

## Two snapshots

Two different input regimes are captured side-by-side so the impact of input
distribution on measured PCC can be tracked. Keep both — they answer different
questions.

### 1. Baseline — `pcc_report.md` / `runxfail.log`

**Inputs**: bf16, uniform `[-1, 1]` on both operands (`run_op_test_with_random_inputs`
with `dtype=torch.bfloat16, minval=-1, maxval=1`).
**Checks**: PCC + allclose (sweeps' `AutomaticValueChecker` mirror).
**Result**: 150/192 fail. Only **13 distinct PCC values**.
**What it measures**: bf16 vs bf16 round-trip consistency. Both device and CPU
golden use bf16, so the gap is small and shows up only as accumulation noise
under specific `optimization_level` + `fp32_dest_acc_en` combos.

### 2. Realistic — `pcc_report_realistic.md` / `runxfail_realistic.log`

**Inputs**: fp32, LLM-style mixture per operand:
* LHS (activations): 99% `N(0, 1)` + 1% `N(0, 10)` — outlier features per Dettmers
  et al. ("LLM.int8: 8-bit Matrix Multiplication for Transformers").
* RHS (weights): same mixture scaled by `1/√K` (Kaiming-style init), so the
  bulk lives in the realistic per-element magnitude for a weight matrix.

**Checks**: PCC only (matches sweeps' effective decision logic via
`check_pcc_error_level`, which classifies purely on PCC ranges and ignores
allclose for xfail).
**Result**: 48/192 fail. Only **3 distinct PCC values** — one per shape.
**What it measures**: bf16 device vs fp32 CPU golden — the real
mixed-precision precision loss sweeps tests. This matches the sweeps numbers
1:1 (e.g. `(32,128,2304)x(2304,1024) FROM_ANOTHER_OP mp_opt2_bf16_fp32accfalse_hifi4`
reports PCC = 0.5004 here, 0.5001 in sweeps — within seed noise).

### Why both regimes

The baseline catches optimizer regressions that happen even in pure bf16
(e.g. fp32 accumulation off changing PCC by 0.02). The realistic regime
catches mixed-precision regressions when the compiler stops promoting through
fp32 for matmul (PCC collapses from ~0.99 to ~0.5).

## Files

- `pcc_report.md` / `pcc_report_realistic.md` — markdown reports (per snapshot).
- `runxfail.log` / `runxfail_realistic.log` — raw pytest logs the reports parse.
- `noforked.log` / `forked.log` — `compare_runs.sh` output (per-test outcome
  sequences from the conftest-fixture vs `--forked` isolation comparison).
- `build_report.py` — turns a `runxfail*.log` into a `pcc_report*.md`.
- `compare_runs.sh` — runs the suite with and without `--forked` and diffs
  the outcome sequences. Confirms `clear_torchxla_computation_cache` provides
  the same isolation as forking.

## Reproducing

The runs assume the sweeps-style venv (e.g. `/localdev/<user>/venv/sweeps/xla`)
and the repo on `PYTHONPATH`. The test currently uses the **realistic inputs**
recipe — to regenerate that snapshot:

```bash
PYTHONPATH="$PWD:$PWD/tests" pytest \
    tests/torch/single_chip/sweeps_derived/test_matmul_mp.py \
    --runxfail --tb=line --no-header -q \
    > tests/torch/single_chip/sweeps_derived/pcc_artifacts/runxfail_realistic.log 2>&1 || true

python3 tests/torch/single_chip/sweeps_derived/pcc_artifacts/build_report.py \
    tests/torch/single_chip/sweeps_derived/pcc_artifacts/runxfail_realistic.log \
    > tests/torch/single_chip/sweeps_derived/pcc_artifacts/pcc_report_realistic.md
```

To regenerate the baseline, the test must temporarily revert to:
`dtype=torch.bfloat16, minval=-1, maxval=1, run_op_test_with_random_inputs(...)`.

To regenerate the forked-vs-noforked comparison:

```bash
PYTHONPATH="$PWD:$PWD/tests" \
    tests/torch/single_chip/sweeps_derived/pcc_artifacts/compare_runs.sh
```

## Current predicate (from realistic snapshot)

`_is_known_pcc_failure(shape_pair, input_source, opt_level, fp32_acc, mf)`:

```
return input_source == "FROM_ANOTHER_OP" and opt_level == 2
```

Covers 48/48 of the realistic-inputs failures. **FROM_HOST + opt=2 passes**
even though `FROM_ANOTHER_OP + opt=2` collapses to PCC ~0.5 — the
`add(x,x)*add(y,y)` prelude is a linear scale on inputs but triggers a
precision-lossy transform at `optimization_level=2` that doesn't fire on the
plain matmul. Worth investigating as a tt-mlir issue.

Within `FROM_ANOTHER_OP + opt=2`, `weight_dtype`, `math_fidelity`, and
`fp32_dest_acc_en` all produce identical PCC to 16 digits — those compiler
options don't appear to reach the matmul kernel in this code path. Same was
true in the baseline snapshot, so it's not specific to the input regime.

## Cross-snapshot findings

- `FROM_ANOTHER_OP` vs `FROM_HOST` give identical PCC in the **baseline**
  (linear scaling argument holds end-to-end in bf16) but differ massively in
  the **realistic** regime (FROM_HOST passes, FROM_ANOTHER_OP collapses). The
  divergence is opt=2 + prelude specific.
- `compare_runs.sh` confirms the conftest cache-clear fixture is
  byte-for-byte equivalent to `--forked` for this suite, at ~2.5× the speed.
