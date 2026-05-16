loader_path: third_party.tt_forge_models.abacusai.causal_lm.pytorch.loader
variant_id: BIGSTRAL_12B_32K
arch: n150
status: DONE_FAIL
test_function: test_abacusai_bigstral_12b_32k
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

# Benchmark added: test_abacusai_bigstral_12b_32k

## Test
tests/benchmark/test_llms.py::test_abacusai_bigstral_12b_32k

## Model
- HF name:    abacusai/bigstral-12b-32k
- Loader:     third_party.tt_forge_models.abacusai.causal_lm.pytorch.loader
- Variant:    ModelVariant.BIGSTRAL_12B_32K ("bigstral_12B_32K")

## Rejection reason

The variant `BIGSTRAL_12B_32K` encodes a **12B** parameter model
(`abacusai/bigstral-12b-32k`). The single-chip n150 capacity ceiling for
LLMs is ≈10B parameters at `bfp_bf8` weights. A 12B model will OOM during
weight transfer regardless of dtype or optimization settings.

Per the skill rules (Step 1.6), models larger than 10B are rejected before
any test code is written or any device time is spent.

No files were modified. No test was added to `tests/benchmark/test_llms.py`.

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
N/A — test not run

## Files changed
- SUMMARY.md (this file only)

## tt-forge-models submodule
no change
