loader_path: third_party.tt_forge_models.devquasar_lfm2_2_6b_transcript_gguf.causal_lm.pytorch.loader
variant_id: 2_6B_Transcript_Q4_K_M_GGUF
arch: p150
status: DONE_FAIL
test_function: test_devquasar_lfm2_2_6b_transcript_gguf
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
failure_reason: "LFM2 hybrid SSM architecture requires Lfm2HybridConvCache (is_compileable=False); harness uses StaticCache with .layers interface; cache type is incompatible with torch.compile/TT-XLA backend"

# Benchmark added: test_devquasar_lfm2_2_6b_transcript_gguf

## Test
tests/benchmark/test_llms.py::test_devquasar_lfm2_2_6b_transcript_gguf

## Model
- HF name:    DevQuasar/LiquidAI.LFM2-2.6B-Transcript-GGUF
- Loader:     third_party.tt_forge_models.devquasar_lfm2_2_6b_transcript_gguf.causal_lm.pytorch.loader
- Variant:    ModelVariant.LFM2_2_6B_TRANSCRIPT_Q4_K_M_GGUF

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

### Error
```
AttributeError: 'StaticCache' object has no attribute 'conv_cache'
  at transformers/models/lfm2/modeling_lfm2.py:522 slow_forward()
```

### Root Cause
LFM2 is a hybrid SSM (State Space Model / Mamba-like) architecture with 30 layers,
24 of which are "conv" (state space) type and 8 are "full_attention" type.

The model requires `Lfm2HybridConvCache` (from transformers) instead of the standard
`StaticCache` used by the benchmark harness. `Lfm2HybridConvCache` has:
- `is_compileable = False` (explicitly marked as incompatible with torch.compile)
- No `.layers` attribute (harness iterates `past_key_values.layers` in transfer_to_device)
- A `conv_cache` attribute for SSM state storage (not present in StaticCache)

### Why This Cannot Be Fixed in This Skill
1. **torch.compile incompatibility**: `Lfm2HybridConvCache.is_compileable = False` means
   the transformers library itself marks this cache as incompatible with torch.compile.
   The TT-XLA benchmark harness requires torch.compile for device execution.
2. **Harness interface mismatch**: The `transfer_to_device` and `_shard_kv_cache` functions
   iterate over `past_key_values.layers`, which `Lfm2HybridConvCache` does not have.
3. **Out of scope**: Supporting hybrid SSM caches for torch.compile would require deep
   changes to the compilation infrastructure (PyTorch/XLA/tt-mlir), not just the test harness.

### Recommended Path Forward
- Add `Lfm2HybridConvCache` support to the benchmark harness (general SSM cache support)
- Or wait for transformers to update `Lfm2HybridConvCache.is_compileable = True` once
  torch.compile compatibility is implemented upstream
- Track as a known incompatibility: hybrid SSM models (Mamba, Jamba, LFM2, etc.) cannot
  currently be benchmarked through the standard causal_lm test_llm harness

## Decode roofline (first decode graph, single-chip)
Source JSON: N/A (test did not reach device execution)
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
- tests/benchmark/test_llms.py (added test_devquasar_lfm2_2_6b_transcript_gguf stub)
- SUMMARY.md

## tt-forge-models submodule
no change — submodule HEAD b671ee900a19 unchanged
