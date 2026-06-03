# PCC artifacts for `test_matmul_mp.py`

Snapshots of measurements and the tooling used to produce them. Regenerate
when the predicate in `test_matmul_mp.py::_is_known_pcc_failure` or any
upstream component (tt-mlir, tt-metal, evaluator thresholds) changes.

## Snapshots

Three input regimes captured side-by-side so the impact of input
distribution on measured PCC can be tracked. The active regime is chosen
via `TTXLA_MATMUL_MP_PROFILE` (default `mixture`); the test ships with
`_mixture_normal` and `_uniform_signed` helpers in place so either can be
re-run without code edits.

### 1. Baseline — `pcc_report.md` / `runxfail.log`

**Inputs**: bf16, uniform `[-1, 1]` on both operands (`run_op_test_with_random_inputs`
with `dtype=torch.bfloat16, minval=-1, maxval=1`).
**Checks**: PCC + allclose (sweeps' `AutomaticValueChecker` mirror).
**Result**: 150/192 fail. Only **13 distinct PCC values**.
**What it measures**: bf16 vs bf16 round-trip consistency. Both device and CPU
golden use bf16, so the gap is small and shows up only as accumulation noise
under specific `optimization_level` + `fp32_dest_acc_en` combos.

### 2. Uniform fp32 — `pcc_report_uniform.md` / `runxfail_uniform.log`

**Inputs**: fp32, uniform `[-1, 1]` on both operands (no Kaiming scaling).
**Run with**: `TTXLA_MATMUL_MP_PROFILE=uniform pytest ...`.
**Result**: 48/192 fail, same exact test_ids as the realistic snapshot.
PCC values within ~0.0005 of the realistic snapshot — distribution shape
doesn't affect PCC once inputs are zero-mean. Atol values are ~5–10× smaller
than the mixture regime because there are no outliers.
**What it measures**: literally what sweeps measures (sweeps generates
`torch.rand * 2 - 1` for `ValueRanges.SMALL`). Use for 1:1 numerical match
against sweeps logs.

### 3. Realistic — `pcc_report_realistic.md` / `runxfail_realistic.log`

**Run with**: `pytest ...` (default profile is `mixture`).
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

### Why all three

- **Baseline** catches optimizer regressions in pure bf16 (e.g. fp32
  accumulation off changing PCC by 0.02).
- **Uniform** is the 1:1 sweeps reproduction: same PCC values to within
  seed noise. Use when chasing parity with a sweeps log.
- **Realistic** is the default test recipe: matches LLM activation/weight
  distributions and surfaces the same precision regressions while giving
  meaningful atol values.

Uniform and Realistic agree on **which tests fail and at what PCC range**.
They differ on **atol magnitudes** because the outlier tail in the mixture
inflates absolute differences. The PCC predicate is invariant across them.

## Files

- `pcc_report.md` / `pcc_report_uniform.md` / `pcc_report_realistic.md` —
  markdown reports (one per snapshot).
- `runxfail.log` / `runxfail_uniform.log` / `runxfail_realistic.log` — raw
  pytest logs the reports parse.
- `sweeps_export.csv` — pivot of the upstream sweeps matmul_mp suite
  (`MIN(pcc)` per `(input_source, shape, run_id)` × 20 columns of
  `(opt, fp32_acc, math_fidelity, weight_dtype)`). Snapshot from the
  sweeps dashboard; see run-timeline table in `findings.md`.
- `compare_with_sweeps_export.py` — cross-checks the test grid against
  `sweeps_export.csv`. Per-shape per-run PCC profile, plus "new
  regressions" and "persistent bugs" buckets.
- `noforked.log` / `forked.log` — `compare_runs.sh` output (per-test outcome
  sequences from the conftest-fixture vs `--forked` isolation comparison).
- `build_report.py` — turns a `runxfail*.log` into a `pcc_report*.md`.
- `compare_runs.sh` — runs the suite with and without `--forked` and diffs
  the outcome sequences. Confirms `clear_torchxla_computation_cache` provides
  the same isolation as forking.

## Reproducing

The runs assume the sweeps-style venv (e.g. `/localdev/<user>/venv/sweeps/xla`)
and the repo on `PYTHONPATH`. Regenerating the realistic snapshot (default
profile):

```bash
PYTHONPATH="$PWD:$PWD/tests" pytest \
    tests/torch/single_chip/sweeps_derived/test_matmul_mp.py \
    --runxfail --tb=line --no-header -q \
    > tests/torch/single_chip/sweeps_derived/pcc_artifacts/runxfail_realistic.log 2>&1 || true

python3 tests/torch/single_chip/sweeps_derived/pcc_artifacts/build_report.py \
    tests/torch/single_chip/sweeps_derived/pcc_artifacts/runxfail_realistic.log \
    > tests/torch/single_chip/sweeps_derived/pcc_artifacts/pcc_report_realistic.md
```

Uniform snapshot — same commands with `TTXLA_MATMUL_MP_PROFILE=uniform` and
the `_uniform` suffixes on the artifacts:

```bash
TTXLA_MATMUL_MP_PROFILE=uniform PYTHONPATH="$PWD:$PWD/tests" pytest \
    tests/torch/single_chip/sweeps_derived/test_matmul_mp.py \
    --runxfail --tb=line --no-header -q \
    > tests/torch/single_chip/sweeps_derived/pcc_artifacts/runxfail_uniform.log 2>&1 || true

python3 tests/torch/single_chip/sweeps_derived/pcc_artifacts/build_report.py \
    tests/torch/single_chip/sweeps_derived/pcc_artifacts/runxfail_uniform.log \
    > tests/torch/single_chip/sweeps_derived/pcc_artifacts/pcc_report_uniform.md
```

To regenerate the bf16-baseline snapshot, the test has to be edited
temporarily to use
`dtype=torch.bfloat16, minval=-1, maxval=1, run_op_test_with_random_inputs(...)`.
That path is not behind `TTXLA_MATMUL_MP_PROFILE` because it represents a
different (round-trip-consistency) test entirely.

To regenerate the forked-vs-noforked comparison:

```bash
PYTHONPATH="$PWD:$PWD/tests" \
    tests/torch/single_chip/sweeps_derived/pcc_artifacts/compare_runs.sh
```

## Current predicate (from realistic snapshot)

`_is_known_pcc_failure(shape_pair, input_source, opt_level, fp32_acc, mf)`:

```
return (
    shape_pair in _FAILING_SHAPES_FOR_PRELUDE_AT_OPT2
    and input_source == "FROM_ANOTHER_OP"
    and opt_level == 2
)
```

Covers 48/48 of the realistic-inputs failures across 4 tested shapes.
**FROM_HOST + opt=2 passes** on every shape — the `add(x,x)*add(y,y)`
prelude is a linear scale on inputs but triggers a precision-lossy transform
at `optimization_level=2` that doesn't fire on the plain matmul.

The collapse is **shape-conditioned**, not universal:

| Shape | FROM_ANOTHER_OP × opt=2 PCC |
|---|---|
| `(32,128,1024)x(1024,2048)` | 0.500 (xfail) |
| `(32,128,2304)x(2304,1024)` | 0.500 (xfail) |
| `(32,128,2560)x(2560,1024)` | 0.500 (xfail) |
| `(32,128,4864)x(4864,896)` | >0.99 (passes) |

Sweeps' xfail txt files agree with this split. The discriminator (N
alignment? K threshold?) is unclear with one passing data point — see
`findings.md` open follow-ups.

Within failing shapes, `weight_dtype`, `math_fidelity`, and
`fp32_dest_acc_en` all produce identical PCC to 16 digits — those compiler
options don't appear to reach the matmul kernel in this code path. Same was
true in the baseline snapshot, so it's not specific to the input regime.

## Cross-snapshot findings

- `FROM_ANOTHER_OP` vs `FROM_HOST` give identical PCC in the **baseline**
  (linear scaling argument holds end-to-end in bf16) but differ massively
  once inputs are fp32 (`uniform` and `realistic`): FROM_HOST passes,
  FROM_ANOTHER_OP collapses. The divergence is opt=2 + prelude specific.
- `uniform` and `realistic` produce **identical failure sets** (same 48
  test_ids, same shape grouping) with PCC values within ~0.0005. Input
  distribution doesn't matter for PCC once inputs are zero-mean; it only
  changes atol scaling. The predicate is the same regardless of the
  active profile.
- `compare_runs.sh` confirms the conftest cache-clear fixture is
  byte-for-byte equivalent to `--forked` for this suite, at ~2.5× the speed.
