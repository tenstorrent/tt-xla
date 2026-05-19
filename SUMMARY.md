loader_path: third_party.tt_forge_models.tiny_gpt2_lm_head.causal_lm.pytorch.loader
variant_id: tiny_gpt2_lm_head
arch: n150
status: DONE_FAIL
test_function: test_tiny_gpt2_lm_head
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
failure_reason: "model config bug: trl-internal-testing/tiny-GPT2LMHeadModel has num_key_value_heads=2 but GPT2 attention always uses num_attention_heads=4 for K/V (no GQA support); benchmark harness init_static_cache initializes StaticCache with 2 heads via config.num_key_value_heads, but actual key_states have 4 heads causing RuntimeError: index_copy_(): Source/destination tensor must have same slice shapes (32 2 2 vs 32 4 2) in CPU reference run"

# Benchmark added: test_tiny_gpt2_lm_head

## Test
tests/benchmark/test_llms.py::test_tiny_gpt2_lm_head

## Model
- HF name:    trl-internal-testing/tiny-GPT2LMHeadModel
- Loader:     third_party.tt_forge_models.tiny_gpt2_lm_head.causal_lm.pytorch.loader
- Variant:    ModelVariant.TINY_GPT2_LM_HEAD

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Failure Analysis
The test fails during the CPU reference run (before any TT device interaction) with:

  RuntimeError: index_copy_(): Source/destination tensor must have same slice shapes.
  Destination slice shape: 32 2 2 at dimension 2 and source slice shape: 32 4 2 at dimension 0.
  (transformers/cache_utils.py:340 inside StaticCache.update)

Root cause: The model `trl-internal-testing/tiny-GPT2LMHeadModel` has an inconsistent config:
- config.num_attention_heads = 4
- config.num_key_value_heads = 2  ← incorrect: GPT2 does NOT implement GQA

GPT2's `c_attn` Conv1D always produces Q/K/V with `num_attention_heads` heads (split_size=embed_dim=8,
reshape to (batch, n_head=4, seq, head_dim=2)). The `num_key_value_heads` config field is unused
by GPT2's attention code.

The benchmark harness (`init_static_cache` in decode_utils.py) reads `config.num_key_value_heads=2`
and passes it to `StaticCache.early_initialization(num_heads=2)`, pre-allocating key/value tensors
with shape (32, 2, max_len, 2). When GPT2 generates key_states of shape (32, 4, seq_len, 2) and
calls `StaticCache.update()`, the shape mismatch causes the crash.

This is a model config bug in the HuggingFace model card. The fix belongs in
`trl-internal-testing/tiny-GPT2LMHeadModel` (setting num_key_value_heads=4 or removing the field).
No modification to third_party/tt_forge_models/ was made.

## Measured (full model, defaults)
- Sample per second:  N/A (DONE_FAIL)
- TTFT (ms):          N/A
- Prefill PCC:        N/A
- First decode PCC:   N/A
- Wall clock:         N/A
- Hardware:           n150

## Decode roofline (first decode graph, single-chip)
Source JSON: N/A (test did not reach device execution)
Achieved vs top_perf_samples_per_sec: N/A

### System
- arch:                        n150 (wormhole_b0)
- chip_count_in_system_desc:   1
- single_chip_assumption:      true
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
- tests/benchmark/test_llms.py (added test_tiny_gpt2_lm_head function)
- SUMMARY.md

## tt-forge-models submodule
no change
