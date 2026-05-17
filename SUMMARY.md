loader_path: third_party.tt_forge_models.bartowski_liquidai_lfm2_2_6b_exp_gguf.causal_lm.pytorch.loader
variant_id: LIQUIDAI_LFM2_2_6B_EXP_Q4_K_M_GGUF
arch: n150
status: DONE_FAIL
test_function: test_liquidai_lfm2_2_6b_exp_gguf
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
failure_reason: "LFM2 model requires Lfm2HybridConvCache (is_compileable=False, dynamic KV cache + conv_cache); benchmark harness uses StaticCache which lacks conv_cache; model architecture (22 SSM/conv + 8 attention layers) incompatible with static-shape device compilation"

# Benchmark added: test_liquidai_lfm2_2_6b_exp_gguf

## Test
tests/benchmark/test_llms.py::test_liquidai_lfm2_2_6b_exp_gguf

## Model
- HF name:    bartowski/LiquidAI_LFM2-2.6B-Exp-GGUF
- Loader:     third_party.tt_forge_models.bartowski_liquidai_lfm2_2_6b_exp_gguf.causal_lm.pytorch.loader
- Variant:    LIQUIDAI_LFM2_2_6B_EXP_Q4_K_M_GGUF

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Failure details

The model failed at the CPU golden-reference step (before any device compilation):

```
AttributeError: 'StaticCache' object has no attribute 'conv_cache'
```

### Root cause

LFM2 is a hybrid SSM/attention model with 30 layers:
- **22 conv/SSM layers** — require `conv_cache` of shape `[batch, hidden_size, 3]`
- **8 full_attention layers** — require standard KV cache

The transformers implementation provides `Lfm2HybridConvCache` which holds both.
The benchmark harness (`construct_inputs` in `llm_benchmark.py`) always creates a
`StaticCache` (attention-only KV cache). Passing it to LFM2's conv layers fails
immediately because `StaticCache` has no `conv_cache` attribute.

Even if the CPU path were fixed with a general `cache_init_fn`, the device
compilation would still fail because:
1. `Lfm2HybridConvCache.is_compileable = False` (explicitly flagged)
2. The KV cache for attention layers is dynamic (concat-based, not pre-allocated)
3. SSM/conv operations are not in the TTNN opset supported by the compiler

### What would be needed to unblock
- A `StaticLfm2HybridCache` that pre-allocates both the attention KV slots and
  the conv state, with in-place updates — and has `is_compileable = True`.
- Verified TTNN support for the convolutional SSM operations used in LFM2.
This work belongs in the tt-forge-models or benchmarking-infra repos.

## Measured (full model, defaults)
- Sample per second:  N/A
- TTFT (ms):          N/A
- Prefill PCC:        N/A
- First decode PCC:   N/A
- Wall clock:         N/A
- Hardware:           n150 (wormhole, single-chip from n300 board)

## Decode roofline (first decode graph, single-chip)
Source JSON: N/A (test did not reach device execution)
Achieved vs top_perf_samples_per_sec: N/A

### System
- arch:                        wormhole_b0
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
- tests/benchmark/test_llms.py (test function added, not runnable yet)

## tt-forge-models submodule
no change
