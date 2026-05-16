loader_path: third_party.tt_forge_models.afrique_gemma_12b_gguf.causal_lm.pytorch.loader
variant_id: AfriqueGemma_12B_Q4_K_M
arch: n150
status: DONE_FAIL
test_function: test_afrique_gemma_12b_q4_k_m
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

# Benchmark added: test_afrique_gemma_12b_q4_k_m

## Test
tests/benchmark/test_llms.py::test_afrique_gemma_12b_q4_k_m

## Model
- HF name:    mradermacher/AfriqueGemma-12B-GGUF
- Loader:     third_party.tt_forge_models.afrique_gemma_12b_gguf.causal_lm.pytorch.loader
- Variant:    AfriqueGemma_12B_Q4_K_M

## Test config landed
- optimization_level:        N/A
- trace_enabled:             N/A
- experimental_weight_dtype: N/A
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  N/A
- TTFT (ms):          N/A
- Prefill PCC:        N/A
- First decode PCC:   N/A
- Wall clock:         N/A
- Hardware:           n150 (n300 board, single chip)

## Decode roofline
Source JSON: N/A (test not run - model exceeds single-chip capacity)

## Files changed
- SUMMARY.md (early-exit report only - no test code added)

## tt-forge-models submodule
no change

## Early-exit reason
AfriqueGemma-12B is a 12B parameter model. Single-chip n150/p150 hardware
can fit at most ~10B parameters at bfp_bf8. Size token 12B detected in variant
id (AfriqueGemma_12B_Q4_K_M) and loader path (afrique_gemma_12b_gguf).
Skipped per Step 1.6 of the add-llm-benchmark-test skill.
