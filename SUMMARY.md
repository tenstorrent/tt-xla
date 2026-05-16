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

## Test config landed
N/A — early exit before test was written

## Measured (full model, defaults)
- Sample per second:  N/A
- TTFT (ms):          N/A
- Prefill PCC:        N/A
- First decode PCC:   N/A
- Wall clock:         N/A
- Hardware:           n150

## Decode roofline (first decode graph, single-chip)
N/A — early exit before any run

## Files changed
- SUMMARY.md (this file)

## tt-forge-models submodule
no change

## Notes
The model `abacusai/bigstral-12b-32k` is a 12B parameter model. The single-chip
n150 capacity ceiling is ~10B parameters at bfp_bf8 weights. This model cannot
fit on a single n150 device and would OOM during weight transfer.

Early exit per Step 1.6: model size ~12B exceeds 10B single-chip capacity.

Source HF runner test:
tests/runner/test_models.py::test_all_models_torch[abacusai/causal_lm/pytorch-bigstral_12B_32K-single_device-inference]
