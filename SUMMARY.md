loader_path: third_party.tt_forge_models.bloomz.causal_lm.pytorch.loader
variant_id: 1B1
arch: p150
status: DONE_FAIL
test_function: test_bloomz_1b1
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
failure_reason: "BLOOM ALiBi attention incompatible with StaticCache benchmark harness: StaticCache.get_seq_length() returns 0, so BloomModel builds attention_mask of prompt length (15) instead of cache length (128), resulting in alibi tensor shape [512, 1, 15] that cannot broadcast with attention scores [512, 15, 128] during CPU golden generation (transformers 5.2.0)"

# Benchmark added: test_bloomz_1b1

## Test
tests/benchmark/test_llms.py::test_bloomz_1b1

## Model
- HF name:    bigscience/bloomz-1b1
- Loader:     third_party.tt_forge_models.bloomz.causal_lm.pytorch.loader
- Variant:    ModelVariant.BLOOMZ_1B1

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
- Hardware:           p150

## Failure Details

The test fails during CPU golden generation (prefill step), before any TT device compilation
occurs. Root cause: BLOOM uses ALiBi (Attention with Linear Biases) positional encoding.
In transformers 5.2.0, `BloomModel.forward()` computes the attention mask as:

    past_length = past_key_values.get_seq_length()  # returns 0 for StaticCache
    seq_length_with_past = prompt_len + past_length  # = 15 + 0 = 15
    attention_mask = torch.ones((batch_size, seq_length_with_past))  # [32, 15]
    alibi = build_alibi_tensor(attention_mask, num_heads)  # [512, 1, 15]

After `StaticCache.update()`, key_layer has shape `[512, head_dim, 128]` (full cache),
so `alibi.baddbmm(query, key)` fails:
    Expected: [512, 15, 128] but alibi has shape [512, 1, 15] — last dim 15 ≠ 128

Fix would require passing a full-length attention_mask (shape [batch, max_cache_len]) to
the model, which requires changes to LLMSamplingWrapper.forward() and construct_inputs()
in the benchmark harness. This is deferred as it requires careful analysis to avoid
breaking other models.

Traceback excerpt:
    RuntimeError: The expanded size of the tensor (128) must match the existing size (15)
    at non-singleton dimension 2. Target sizes: [512, 15, 128]. Tensor sizes: [512, 1, 15]

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

## Files changed
- tests/benchmark/test_llms.py (test_bloomz_1b1 added at line 1094)

## tt-forge-models submodule
no change
