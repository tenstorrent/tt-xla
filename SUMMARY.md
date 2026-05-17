loader_path: third_party.tt_forge_models.bielik_11b_v2_3_instruct_gguf.causal_lm.pytorch.loader
variant_id: 11B_v2.3_Instruct_GGUF
arch: n150
status: DONE_FAIL
test_function: test_bielik_11b_v2_3_instruct_gguf
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
failure_reason: "model size ~11B exceeds 10B single-chip capacity"

# Benchmark added: test_bielik_11b_v2_3_instruct_gguf

## Test
tests/benchmark/test_llms.py::test_bielik_11b_v2_3_instruct_gguf

## Model
- HF name:    speakleash/Bielik-11B-v2.3-Instruct-GGUF
- Loader:     third_party.tt_forge_models.bielik_11b_v2_3_instruct_gguf.causal_lm.pytorch.loader
- Variant:    ModelVariant.BIELIK_11B_V2_3_INSTRUCT_GGUF ("11B_v2.3_Instruct_GGUF")

## Early Exit Reason

The variant `11B_v2.3_Instruct_GGUF` contains size token `11B` (> 10B).
Single-chip n150/p150 cannot fit ~11B-parameter models even at `bfp_bf8` weights —
the model will OOM during weight transfer. Per Step 1.6 of the skill, this model
is rejected up front without attempting a test run.

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
- Hardware:           n150 (Wormhole n300 board)

## Decode roofline (first decode graph, single-chip)
Source JSON: N/A
Achieved vs top_perf_samples_per_sec: N/A

## Files changed
- SUMMARY.md (only — no test added, no code changes)

## tt-forge-models submodule
no change
