loader_path: third_party.tt_forge_models.codellama_13b_python_gguf.causal_lm.pytorch.loader
variant_id: CodeLlama_13B_Python_Q4_K_M_GGUF
arch: n150
status: DONE_FAIL
test_function: test_codellama_13b_python_gguf
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
failure_reason: "model size ~13B exceeds 10B single-chip capacity"

# Benchmark added: test_codellama_13b_python_gguf

## Test
tests/benchmark/test_llms.py::test_codellama_13b_python_gguf

## Model
- HF name:    codellama/CodeLlama-13b-Python-hf (GGUF: codellama-13b-python.Q4_K_M.gguf)
- Loader:     third_party.tt_forge_models.codellama_13b_python_gguf.causal_lm.pytorch.loader
- Variant:    CodeLlama_13B_Python_Q4_K_M_GGUF

## Test config landed
N/A — early exit before test was written

## Measured (full model, defaults)
N/A — early exit before test was run

## Decode roofline (first decode graph, single-chip)
N/A — early exit before test was run

## Files changed
None — no test written (early exit at Step 1.6)

## tt-forge-models submodule
no change
