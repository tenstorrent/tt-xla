loader_path: third_party.tt_forge_models.afrique_gemma_12b_gguf.causal_lm.pytorch.loader
variant_id: AfriqueGemma_12B_Q4_K_M
arch: n150
status: DONE_FAIL
test_function: test_afrique_gemma_12b_gguf
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

# Benchmark added: test_afrique_gemma_12b_gguf

## Test
tests/benchmark/test_llms.py::test_afrique_gemma_12b_gguf

## Model
- HF name:    mradermacher/AfriqueGemma-12B-GGUF
- Loader:     third_party.tt_forge_models.afrique_gemma_12b_gguf.causal_lm.pytorch.loader
- Variant:    AfriqueGemma_12B_Q4_K_M

## Test config landed
N/A — early exit before test was added (model size exceeds single-chip capacity)

## Measured (full model, defaults)
N/A

## Decode roofline (first decode graph, single-chip)
N/A

## Files changed
None — no test was added

## tt-forge-models submodule
no change
