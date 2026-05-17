loader_path: third_party.tt_forge_models.carbon_villain.causal_lm.pytorch.loader
variant_id: CarbonVillain_en_10.7B_v4
arch: n150
status: DONE_FAIL
test_function: test_carbon_villain_en_10_7b_v4
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
failure_reason: "model size ~10.7B exceeds 10B single-chip capacity"

# Benchmark added: test_carbon_villain_en_10_7b_v4

## Test
tests/benchmark/test_llms.py::test_carbon_villain_en_10_7b_v4

## Model
- HF name:    jeonsworld/CarbonVillain-en-10.7B-v4
- Loader:     third_party.tt_forge_models.carbon_villain.causal_lm.pytorch.loader
- Variant:    CarbonVillain_en_10.7B_v4

## Test config landed
- optimization_level:        N/A
- trace_enabled:             N/A
- experimental_weight_dtype: N/A
- batch_size:                N/A
- input_sequence_length:     N/A
- required_pcc:              N/A

## Measured (full model, defaults)
- Sample per second:  N/A
- TTFT (ms):          N/A
- Prefill PCC:        N/A
- First decode PCC:   N/A
- Wall clock:         N/A
- Hardware:           n150

## Decode roofline (first decode graph, single-chip)
Source JSON: N/A
Achieved vs top_perf_samples_per_sec: N/A

Early exit: model size ~10.7B exceeds 10B single-chip capacity ceiling.
The variant CarbonVillain_en_10.7B_v4 (jeonsworld/CarbonVillain-en-10.7B-v4)
has 10.7B parameters which exceeds the ~10B parameter limit for n150 single-chip
at bfp_bf8 weights. Writing a test and running it would result in OOM during
weight transfer.

## Files changed
(none — early exit before test was written)

## tt-forge-models submodule
no change
