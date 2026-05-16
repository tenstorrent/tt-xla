loader_path: third_party.tt_forge_models.afrique_qwen.causal_lm.pytorch.loader
variant_id: 14B
arch: n150
status: DONE_FAIL
test_function: test_afrique_qwen_14b
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
failure_reason: "model size ~14B exceeds 10B single-chip capacity"

# Benchmark skipped: test_afrique_qwen_14b

## Test
tests/benchmark/test_llms.py::test_afrique_qwen_14b

## Model
- HF name:    McGill-NLP/AfriqueQwen-14B
- Loader:     third_party.tt_forge_models.afrique_qwen.causal_lm.pytorch.loader
- Variant:    ModelVariant.AFRIQUE_QWEN_14B ("14B")

## Rejection reason (Step 1.6)
The variant `14B` matches the size-token pattern `[0-9]+(\.[0-9]+)?[bB]` with
a size of **14B**, which exceeds the 10B single-chip capacity ceiling for n150
(Wormhole). The model would OOM during weight transfer regardless of dtype or
optimization knobs.

The loader at submodule HEAD (2a90c3003b) **does** contain `AFRIQUE_QWEN_14B`
in its `ModelVariant` enum, so the variant itself is not missing — the model
is simply too large for a single n150/p150 chip at any supported dtype.

If this model needs to be benchmarked it would require a multi-chip (TP) setup,
which is out of scope for the single-chip `test_llm` harness used by this skill.

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
- Hardware:           n300 (Wormhole, single-chip path)

## Decode roofline (first decode graph, single-chip)
N/A — test not run

## Files changed
- SUMMARY.md (this file, early-exit record)

## tt-forge-models submodule
no change
