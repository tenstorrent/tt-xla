loader_path: third_party.tt_forge_models.70b_incisive_vernacular_gguf.causal_lm.pytorch.loader
variant_id: INCISIVE_VERNACULAR_70B_Q4_K_M_GGUF
arch: n150
status: DONE_FAIL
test_function: test_70b_incisive_vernacular_q4_k_m_gguf
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
failure_reason: "model size ~70B exceeds 10B single-chip capacity"

# Benchmark added: test_70b_incisive_vernacular_q4_k_m_gguf

## Test
tests/benchmark/test_llms.py::test_70b_incisive_vernacular_q4_k_m_gguf

## Model
- HF name:    mradermacher/70B_Incisive_Vernacular-i1-GGUF
- Loader:     third_party.tt_forge_models.70b_incisive_vernacular_gguf.causal_lm.pytorch.loader
- Variant:    INCISIVE_VERNACULAR_70B_Q4_K_M_GGUF (value: "70B_INCISIVE_VERNACULAR_Q4_K_M_GGUF")

## Early Exit Reason
Model size ~70B far exceeds the single-chip n150/p150 capacity ceiling of ~10B parameters even
at bfp_bf8 weights. The variant name `70B_INCISIVE_VERNACULAR_Q4_K_M_GGUF` and the HuggingFace
repo `mradermacher/70B_Incisive_Vernacular-i1-GGUF` both confirm the 70B scale. No test was
added; no benchmark run was attempted.

## Test config landed
N/A — no test added

## Measured (full model, defaults)
- Sample per second:  N/A
- TTFT (ms):          N/A
- Prefill PCC:        N/A
- First decode PCC:   N/A
- Wall clock:         N/A
- Hardware:           n150 (Wormhole, n300 board)

## Decode roofline (first decode graph, single-chip)
N/A — no run attempted

## Files changed
- SUMMARY.md (this file only — no test code added)

## tt-forge-models submodule
no change
