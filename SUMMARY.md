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
- Variant:    BIGSTRAL_12B_32K (= "bigstral_12B_32K")

## Early-exit reason
The variant name and `pretrained_model_name` both indicate a **12B** parameter
model. The single-chip n150 (Wormhole_b0) can fit at most ≈8 B parameters even
at bfp_bf8 weights. 12 B > 10 B threshold → rejected up front per Step 1.6 of
the add-llm-benchmark-test skill to avoid wasting a device hour on a guaranteed
OOM.

If this model should run on a multi-chip or p150 setup (≲25 B threshold),
re-run the skill with the appropriate TP harness and target hardware.

## Test config landed
N/A — test not added (early exit)

## Measured (full model, defaults)
N/A

## Decode roofline (first decode graph, single-chip)
N/A

## Files changed
- SUMMARY.md (this file)

## tt-forge-models submodule
no change
