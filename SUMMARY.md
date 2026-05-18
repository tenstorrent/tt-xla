loader_path: third_party.tt_forge_models.dans_personality_engine_gguf.causal_lm.pytorch.loader
variant_id: V1_3_0_12B_GGUF
arch: n150
status: DONE_FAIL
test_function: test_dans_personality_engine_v1_3_0_12b_gguf
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

# Benchmark added: test_dans_personality_engine_v1_3_0_12b_gguf

## Test
tests/benchmark/test_llms.py::test_dans_personality_engine_v1_3_0_12b_gguf

## Model
- HF name:    bartowski/PocketDoc_Dans-PersonalityEngine-V1.3.0-12b-GGUF
- Loader:     third_party.tt_forge_models.dans_personality_engine_gguf.causal_lm.pytorch.loader
- Variant:    ModelVariant.DANS_PERSONALITY_ENGINE_V1_3_0_12B_GGUF (V1_3_0_12B_GGUF)

## Early Exit: Model Too Large

The variant `V1_3_0_12B_GGUF` maps to a ~12B parameter model
(`bartowski/PocketDoc_Dans-PersonalityEngine-V1.3.0-12b-GGUF`).

The 10B single-chip capacity cutoff for n150/p150 (even at bfp_bf8 weights)
means this model will OOM during weight transfer. Per Step 1.6 of the skill,
models with a largest size token > 10B are rejected at this stage rather than
spending device time on a doomed run.

Size evidence:
- Variant id:             V1_3_0_12B_GGUF  → "12B"
- pretrained_model_name:  …-V1.3.0-12b-GGUF → "12b"

## Test config landed
- optimization_level:        N/A (not added)
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
- Hardware:           n150 (Wormhole n300 board, single die)

## Decode roofline (first decode graph, single-chip)
Source JSON: N/A
Achieved vs top_perf_samples_per_sec: N/A

## Files changed
- SUMMARY.md (this file only — no test was added)

## tt-forge-models submodule
no change
