loader_path: third_party.tt_forge_models.lfm2_5_1_2b_thinking_kimi_v2_distill_gguf.causal_lm.pytorch.loader
variant_id: LFM2_5_1_2B_THINKING_KIMI_V2_DISTILL_Q4_K_M_GGUF
arch: p150
status: DONE_FAIL
test_function: test_lfm2_5_1_2b_thinking_kimi_v2_distill_gguf
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
failure_reason: "LFM2 model uses Lfm2HybridConvCache (hybrid SSM/attention cache with conv_cache attribute) which is incompatible with the StaticCache used by the benchmark harness; AttributeError: 'StaticCache' object has no attribute 'conv_cache' during CPU golden forward pass"

# Benchmark added: test_lfm2_5_1_2b_thinking_kimi_v2_distill_gguf

## Test
tests/benchmark/test_llms.py::test_lfm2_5_1_2b_thinking_kimi_v2_distill_gguf

## Model
- HF name:    mradermacher/LFM2.5-1.2B-Thinking-Kimi-V2-DISTILL-GGUF
- Loader:     third_party.tt_forge_models.lfm2_5_1_2b_thinking_kimi_v2_distill_gguf.causal_lm.pytorch.loader
- Variant:    LFM2_5_1_2B_THINKING_KIMI_V2_DISTILL_Q4_K_M_GGUF

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: bfp_bf8
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  N/A
- TTFT (ms):          N/A
- Prefill PCC:        N/A
- First decode PCC:   N/A
- Wall clock:         N/A
- Hardware:           p150

## Failure Details
The LFM2 model (Liquid Foundation Model 2) uses a hybrid SSM/attention architecture
that requires `Lfm2HybridConvCache` — a custom cache class with both `conv_cache`
(for SSM/convolutional recurrence layers) and standard KV cache (for attention layers).

The benchmark harness (`llm_benchmark.py`) creates and passes a `StaticCache` object,
which is incompatible: `StaticCache` has no `conv_cache` attribute and does not
implement the `Lfm2HybridConvCache` interface.

Error (CPU golden forward pass):
  AttributeError: 'StaticCache' object has no attribute 'conv_cache'
  at transformers/models/lfm2/modeling_lfm2.py:522 in slow_forward()

Fixing this would require the harness to support model-specific hybrid caches,
which is out of scope for this skill. The fix belongs in the benchmark infrastructure
as a general "hybrid SSM/attention model" support path.

## Decode roofline (first decode graph, single-chip)
Source JSON: N/A
Achieved vs top_perf_samples_per_sec: N/A

### System
- arch:                        p150
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

## Files changed
- tests/benchmark/test_llms.py

## tt-forge-models submodule
no change
