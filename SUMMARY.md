loader_path: third_party.tt_forge_models.deepseek_r1_distill_qwen_14b_gguf.causal_lm.pytorch.loader
variant_id: DeepSeek_R1_Distill_Qwen_14B_GGUF
arch: n150
status: DONE_FAIL
test_function: test_deepseek_r1_distill_qwen_14b_gguf
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

# Benchmark added: test_deepseek_r1_distill_qwen_14b_gguf

## Test
tests/benchmark/test_llms.py::test_deepseek_r1_distill_qwen_14b_gguf

## Model
- HF name:    osllmai-community/DeepSeek-R1-Distill-Qwen-14B-GGUF
- Loader:     third_party.tt_forge_models.deepseek_r1_distill_qwen_14b_gguf.causal_lm.pytorch.loader
- Variant:    DeepSeek_R1_Distill_Qwen_14B_GGUF

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
Source JSON: N/A
Achieved vs top_perf_samples_per_sec: N/A

Early exit: model size ~14B exceeds single-chip n150/p150 capacity (≲10B params limit).
The size token "14B" was found in the variant id (DeepSeek_R1_Distill_Qwen_14B_GGUF) and
confirmed in the loader's pretrained_model_name
(osllmai-community/DeepSeek-R1-Distill-Qwen-14B-GGUF). No test code was added and no
files under third_party/tt_forge_models/ were modified.

## Files changed
- SUMMARY.md (this file)

## tt-forge-models submodule
no change
