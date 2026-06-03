# PCC artifacts for `test_matmul_mp.py`

Snapshot of measurements and the tooling used to produce them. Regenerate
when the predicate in `test_matmul_mp.py::_is_known_pcc_failure` or any
upstream component (tt-mlir, tt-metal, evaluator thresholds) changes.

## Files

- `pcc_report.md` — markdown summary: failures grouped by `(shape, opt,
  fp32_dest_acc_en, weight_dtype, math_fidelity)`, with PCC values pivoted
  on input source.
- `runxfail.log` — pytest log captured with `--runxfail --tb=line`. Every
  xfail-marked case runs to completion and prints its `Calculated: pcc=...`
  AssertionError, which is what the report consumes.
- `noforked.log` / `forked.log` — per-test outcome sequences from the
  comparison run.
- `build_report.py` — turns `runxfail.log` into `pcc_report.md`.
- `compare_runs.sh` — runs the suite with and without `--forked` and diffs
  the outcome sequences. Used to confirm that the conftest
  `clear_torchxla_computation_cache` fixture provides the same isolation
  as forking.

## Reproducing

The runs assume the sweeps-style venv (the regular one used by sweeps
operators, e.g. `/localdev/<user>/venv/sweeps/xla`) and the repo on
`PYTHONPATH`. From the repo root:

```bash
# 1. Regenerate runxfail.log + pcc_report.md
PYTHONPATH="$PWD:$PWD/tests" pytest \
    tests/torch/single_chip/sweeps_derived/test_matmul_mp.py \
    --runxfail --tb=line --no-header -q \
    > tests/torch/single_chip/sweeps_derived/pcc_artifacts/runxfail.log 2>&1 || true

python3 tests/torch/single_chip/sweeps_derived/pcc_artifacts/build_report.py \
    tests/torch/single_chip/sweeps_derived/pcc_artifacts/runxfail.log \
    > tests/torch/single_chip/sweeps_derived/pcc_artifacts/pcc_report.md

# 2. Regenerate forked vs noforked comparison
PYTHONPATH="$PWD:$PWD/tests" \
    tests/torch/single_chip/sweeps_derived/pcc_artifacts/compare_runs.sh
```

## Notable findings (from the snapshot)

- 150/192 cases fail PCC; only 13 distinct PCC values across them.
- PCC is fully determined by `(shape, opt_level, fp32_dest_acc_en)`. Within
  one such bucket, `weight_dtype` and `math_fidelity` produce identical PCC
  to 16 digits — strongly suggests those two compiler options don't reach
  the matmul kernel in this path. Worth a follow-up investigation.
- `FROM_ANOTHER_OP` and `FROM_HOST` give identical PCC for every combo
  (the `add(x,x)*add(y,y) = 4·x·y` prelude is a linear scaling).
- `compare_runs.sh` confirms the conftest cache-clear fixture is byte-for-byte
  equivalent to `--forked` for this suite, at ~2.5× the speed.
