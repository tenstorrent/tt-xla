loader_path: third_party.tt_forge_models.captain_eris_violet_gguf.causal_lm.pytorch.loader
variant_id: quantfactory_V0.420_12B_GGUF
arch: n150
status: DONE_FAIL
test_function: test_captain_eris_violet_gguf
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
failure_reason: "model size ~12B exceeds 10B single-chip capacity"

# Benchmark added: test_captain_eris_violet_gguf

## Test
tests/benchmark/test_llms.py::test_captain_eris_violet_gguf

## Model
- HF name:    QuantFactory/Captain-Eris_Violet-V0.420-12B-GGUF
- Loader:     third_party.tt_forge_models.captain_eris_violet_gguf.causal_lm.pytorch.loader
- Variant:    quantfactory_V0.420_12B_GGUF

## Test config landed
- optimization_level:        N/A
- trace_enabled:             N/A
- experimental_weight_dtype: N/A
- batch_size:                N/A
- input_sequence_length:     N/A
- required_pcc:              N/A

## Early exit reason
Model size ~12B exceeds 10B single-chip capacity limit (Step 1.6).
The variant `quantfactory_V0.420_12B_GGUF` and the pretrained model name
`QuantFactory/Captain-Eris_Violet-V0.420-12B-GGUF` both contain the size token
`12B`, which exceeds the ≲10B parameter ceiling for a single n150/p150 chip at
bfp_bf8 weights. Attempting to run would OOM during weight transfer.

## Measured (full model, defaults)
- Sample per second:  N/A
- TTFT (ms):          N/A
- Prefill PCC:        N/A
- First decode PCC:   N/A
- Wall clock:         N/A
- Hardware:           n150 (n300 L chip)

## Decode roofline (first decode graph, single-chip)
N/A — test not executed

## Files changed
- SUMMARY.md (only)

## tt-forge-models submodule
no change
