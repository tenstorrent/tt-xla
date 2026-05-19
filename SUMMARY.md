loader_path: third_party.tt_forge_models.inclusionai_asearcher_local_14b.causal_lm.pytorch.loader
variant_id: ASearcher_Local_14B
arch: n150
status: DONE_FAIL
test_function: test_inclusionai_asearcher_local_14b
samples_per_second: null
ttft_ms: null
prefill_pcc: null
first_decode_pcc: null
top_perf_samples_per_sec: null
pct_of_target: null
roofline_bound: null
optimization_level: null
trace_enabled: null
experimental_weight_dtype: null
failure_reason: "model size ~14B exceeds 10B single-chip capacity on n150"

# Benchmark added: test_inclusionai_asearcher_local_14b

## Test
tests/benchmark/test_llms.py::test_inclusionai_asearcher_local_14b

## Model
- HF name:    inclusionAI/ASearcher-Local-14B
- Loader:     third_party.tt_forge_models.inclusionai_asearcher_local_14b.causal_lm.pytorch.loader
- Variant:    ModelVariant.ASEARCHER_LOCAL_14B

## Test config landed
N/A — early exit before test was added (model exceeds single-chip capacity)

## Early Exit Reason
The variant ASearcher_Local_14B is a 14B parameter model. On n150 (wormhole_b0),
the single-chip capacity cap is 10B parameters. At 14B, the model will OOM during
weight transfer regardless of dtype overrides. No test was added to test_llms.py.

## Measured (full model, defaults)
- Sample per second:  N/A
- TTFT (ms):          N/A
- Prefill PCC:        N/A
- First decode PCC:   N/A
- Wall clock:         N/A
- Hardware:           n150

## Decode roofline (first decode graph, single-chip)
N/A — early exit

## Files changed
- SUMMARY.md (this file only — no test added)

## tt-forge-models submodule
no change (submodule HEAD: 6c39ebb532)
