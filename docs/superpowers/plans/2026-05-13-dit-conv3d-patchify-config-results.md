# DiT Conv3d Patchify Heuristic — Measurement Results

## Default-config baseline (Python patch disabled, tt-mlir unchanged)

- Conv3d per-op kernel duration (mean of N invocations): **109.212 ms** (N=28, across 7 forward passes × 4 chips; identical input shape W=21, Z=60, Y=104, X=32; range 109.024–109.388 ms; median 109.273 ms)
- DiT e2e step time (PERF_MODE, MAX_BLOCKS=1, 480p sharded, mean of warm runs): **1357.264 ms** (mean over 4 warm runs after excluding a 2684.97 ms Tracy stall outlier; raw warm times: 1360.2489, 1353.8040, 2684.9685, 1364.0007, 1351.0049 ms)
- PCC: **0.9994683861732483**

Source log: `/tmp/dit_default_config_baseline.log`
Source Tracy CSV: `/root/tt-xla/.tracy_artifacts/reports/2026_05_13_08_03_15/ops_perf_results_2026_05_13_08_03_15.csv` (originally generated at `/root/tt-metal/generated/profiler/reports/2026_05_13_08_03_15/`)

## Python-patched baseline (reference from `test_wan_dit.py` docstring)

- DiT e2e step time: 1292 ms
- PCC: 0.99941

## Heuristic candidate (Python patch disabled, tt-mlir change applied)

- Conv3d per-op kernel duration (mean of N): **11.337 ms** (N=28, median 11.317 ms, min 11.181 ms, max 11.534 ms, FPU util mean 2.108 %). All rows share identical shape W=21, Z=60, Y=104, X=32 (same as the default-config baseline). 32 Conv3d rows present in CSV; 4 had NaN kernel-duration entries (host-side artifacts from a different run column), so N=28 valid device measurements — matching the baseline N=28.
- DiT e2e step time (warm runs, dropping a 2600.26 ms Tracy stall outlier): **1250.054 ms** (mean over 4 warm runs after excluding the outlier; raw warm times: 1245.9529, 1235.7941, 2600.2645, 1266.0941, 1252.3754 ms; median of filtered 1249.164 ms)
- PCC: **0.9994683861732483** (identical to default-config baseline)

Source log: `/tmp/dit_heuristic_candidate.log`
Source Tracy CSV: `/root/tt-metal/generated/profiler/reports/2026_05_13_08_21_02/ops_perf_results_2026_05_13_08_21_02.csv`

## Measurement-based acceptance evaluation

| Gate | Threshold | Measured | Pass? |
|---|---|---|---|
| PCC | >= 0.9994683861732483 (default-config baseline) | 0.9994683861732483 | YES |
| Per-op kernel >= 5x faster than default-config baseline | <= 21.842 ms | 11.337 ms (9.63x faster) | YES |
| E2e step within 5% of Python-patched baseline (1292 ms) | <= 1356.6 ms | 1250.054 ms (-3.25 %, i.e. 1.03x faster than 1292 ms reference) | YES |

**Recommendation to controller (no decision authority, just observation):** based on the measured numbers vs gates, the candidate state appears to pass all three gates cleanly (not borderline on any). PCC is bit-identical to the default-config baseline. Conv3d kernel speedup is ~9.6x, comfortably beyond the 5x bar. E2e step time is actually *better* than the Python-patched docstring reference of 1292 ms by ~3.25 %, well inside the 5 % tolerance. No run-to-run-noise concern flagged for any gate.
