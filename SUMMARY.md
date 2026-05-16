loader_path: third_party.tt_forge_models.70b_incisive_vernacular_gguf.causal_lm.pytorch.loader
variant_id: INCISIVE_VERNACULAR_70B_Q4_K_M_GGUF
arch: n150
status: DONE_FAIL
test_function: test_70b_incisive_vernacular_gguf
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

# Benchmark added: test_70b_incisive_vernacular_gguf

## Test
tests/benchmark/test_llms.py::test_70b_incisive_vernacular_gguf

## Model
- HF name:    mradermacher/70B_Incisive_Vernacular-i1-GGUF
- Loader:     third_party.tt_forge_models.70b_incisive_vernacular_gguf.causal_lm.pytorch.loader
- Variant:    ModelVariant.INCISIVE_VERNACULAR_70B_Q4_K_M_GGUF = "70B_INCISIVE_VERNACULAR_Q4_K_M_GGUF"

## Rejection reason (Step 1.6)

The variant name `70B_INCISIVE_VERNACULAR_Q4_K_M_GGUF` contains the size token `70B`.
At 70B parameters, the model cannot fit in a single n150/p150 chip's DRAM even at `bfp_bf8`
weights — the single-chip capacity ceiling is ≲ 10B parameters. Writing a test and
downloading the GGUF (which is multi-GB) would OOM unconditionally.

This model requires a multi-chip (TP) harness (`test_llm_tp`) and a galaxy/n300 target,
which is out of scope for this single-chip benchmark skill.

## Test config landed
- optimization_level:        N/A (test not written)
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
- Hardware:           n300 (Wormhole, 2× n150 chips)

## Decode roofline (first decode graph, single-chip)
Source JSON: N/A
Achieved vs top_perf_samples_per_sec: N/A

## Files changed
- SUMMARY.md (this file only — no test was written)

## tt-forge-models submodule
no change — variant INCISIVE_VERNACULAR_70B_Q4_K_M_GGUF present at submodule HEAD fd5a61806e
(loader added in 0acfcea7d6, gguf dep fixed in 833971aa18)
