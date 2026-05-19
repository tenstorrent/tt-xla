loader_path: third_party.tt_forge_models.glm_4_1v_gguf.causal_lm.pytorch.loader
variant_id: 4.1V_9B_Thinking_GGUF
arch: p150
status: DONE_FAIL
test_function: test_glm_4_1v_9b_thinking_gguf
samples_per_second: null
ttft_ms: null
prefill_pcc: null
first_decode_pcc: null
top_perf_samples_per_sec: null
pct_of_target: null
roofline_bound: null
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: bfp_bf8
failure_reason: "GGUF model with architecture glm4 is not supported by installed transformers version (transformers==5.2.0): ValueError: GGUF model with architecture glm4 is not supported yet."

# Benchmark added: test_glm_4_1v_9b_thinking_gguf

## Test
tests/benchmark/test_llms.py::test_glm_4_1v_9b_thinking_gguf

## Model
- HF name:    unsloth/GLM-4.1V-9B-Thinking-GGUF
- Loader:     third_party.tt_forge_models.glm_4_1v_gguf.causal_lm.pytorch.loader
- Variant:    ModelVariant.GLM_4_1V_9B_THINKING_GGUF ("4.1V_9B_Thinking_GGUF")

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  N/A (test failed at model load)
- TTFT (ms):          N/A
- Prefill PCC:        N/A
- First decode PCC:   N/A
- Wall clock:         N/A
- Hardware:           p150

## Failure Details
The model failed to load during the tokenizer/model initialization phase with:

    ValueError: GGUF model with architecture glm4 is not supported yet.

This error originates from `transformers.modeling_gguf_pytorch_utils.load_gguf_checkpoint`
(transformers==5.2.0). The GLM-4.1V model uses the `glm4` GGUF architecture which is not
yet supported by the transformers version in this environment. This is a library compatibility
issue and cannot be resolved within the scope of this benchmark skill — it requires either:
1. An upgrade to a transformers version that supports `glm4` GGUF format.
2. A fix in the model loader to handle loading via a different mechanism.

## Decode roofline (first decode graph, single-chip)
Source JSON: N/A (test failed before execution)
Achieved vs top_perf_samples_per_sec: N/A

### System
- arch: N/A
- chip_count_in_system_desc: N/A
- single_chip_assumption: N/A
- worker_grid_cores: N/A
- dram_bandwidth_bytes_per_sec: N/A

### Roofline
- bound: N/A
- top_perf_samples_per_sec: N/A
- top_perf_time_ms: N/A

## Files changed
- tests/benchmark/test_llms.py (test function added)

## tt-forge-models submodule
no change
