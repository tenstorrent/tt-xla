loader_path: third_party.tt_forge_models.llada_2.causal_lm.pytorch.loader
variant_id: llada_2_0_mini
arch: p150
status: DONE_FAIL
test_function: test_llada_2_0_mini
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
failure_reason: "AttributeError: 'NoneType' object has no attribute 'size' in modeling_llada2_moe.py:866 — model forward crashes when attention_mask=None; model uses trust_remote_code with custom forward that does not handle absent attention_mask in decode harness"

# Benchmark added: llada_2_0_mini

## Test
tests/benchmark/test_llms.py::test_llada_2_0_mini

## Model
- HF name:    inclusionAI/LLaDA2.0-mini
- Loader:     third_party.tt_forge_models.llada_2.causal_lm.pytorch.loader
- Variant:    ModelVariant.LLADA_2_0_MINI

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  N/A
- TTFT (ms):          N/A
- Prefill PCC:        N/A
- First decode PCC:   N/A
- Wall clock:         N/A
- Hardware:           p150 (Blackhole p300c)

## Decode roofline (first decode graph, single-chip)
Source JSON: N/A
Achieved vs top_perf_samples_per_sec: N/A

### System
- arch:                        N/A
- chip_count_in_system_desc:   N/A
- single_chip_assumption:      N/A
- worker_grid_cores:           N/A
- dram_bandwidth_bytes_per_sec: N/A

### Peak FLOPs
- lofi:  N/A
- hifi2: N/A
- hifi3: N/A
- hifi4: N/A

### Compute
- total_flops:             N/A
- breakdown.matmul:        N/A
- breakdown.linear:        N/A
- breakdown.conv2d:        N/A
- breakdown.sparse_matmul: N/A

### Inputs
- count:        N/A
- memory_bytes: N/A

### KV cache
- count:        N/A
- memory_bytes: N/A
- memory_gb:    N/A

### Params
- count:                  N/A
- effective_count:        N/A
- memory_bytes:           N/A
- memory_gb:              N/A
- effective_memory_bytes: N/A
- effective_memory_gb:    N/A
- embedding_count:        N/A
- embedding_memory_bytes: N/A

### Roofline
- bound:                    N/A
- top_perf_samples_per_sec: N/A
- top_perf_time_ms:         N/A
- dram_time_ms:             N/A
- compute_time_ms_lofi:     N/A
- compute_time_ms_hifi2:    N/A
- compute_time_ms_hifi3:    N/A
- compute_time_ms_hifi4:    N/A

## Failure details
The test fails during the CPU prefill step with:

    AttributeError: 'NoneType' object has no attribute 'size'

in `modeling_llada2_moe.py:866`:

    if attention_mask.size() == (batch_size, 1, seq_length, seq_length):

The model uses `trust_remote_code=True` and its custom `LLaDA2MoeModel.forward` does not handle `attention_mask=None`. The benchmark decode harness (`LLMSamplingWrapper.forward`) intentionally does not pass `attention_mask` (matching standard HuggingFace convention), but this model's remote code requires an explicit 4D causal mask. Fixing this requires patching the model's forward in `third_party/tt_forge_models/`, which is out of scope for this skill.

## Files changed
- tests/benchmark/test_llms.py
- .github/workflows/perf-bench-matrix.json

## tt-forge-models submodule
no change
