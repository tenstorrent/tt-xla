loader_path: third_party.tt_forge_models.granite_4_0_1b_gguf.causal_lm.pytorch.loader
variant_id: Granite_4.0_1B_Q4_K_M
arch: p150
status: DONE_FAIL
test_function: test_granite_4_0_1b_q4_k_m
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
failure_reason: "model forward raises AttributeError: 'StaticCache' object has no attribute 'has_previous_state' in transformers 5.2.0 granitemoehybrid modeling — Mamba-specific attribute accessed on StaticCache; model definition bug, not fixable in test"

# Benchmark added: test_granite_4_0_1b_q4_k_m

## Test
tests/benchmark/test_llms.py::test_granite_4_0_1b_q4_k_m

## Model
- HF name:    ibm-granite/granite-4.0-1b-base
- Loader:     third_party.tt_forge_models.granite_4_0_1b_gguf.causal_lm.pytorch.loader
- Variant:    ModelVariant.GRANITE_4_0_1B_Q4_K_M ("Granite_4.0_1B_Q4_K_M")

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

## Failure
The model failed with:

```
AttributeError: 'StaticCache' object has no attribute 'has_previous_state'
```

in `venv/lib/python3.12/site-packages/transformers/models/granitemoehybrid/modeling_granitemoehybrid.py:1334`:

```python
if past_key_values and not past_key_values.has_previous_state:
```

The `granitemoehybrid` forward method accesses `has_previous_state` on the KV cache — an attribute that only exists on `MambaCache`/`HybridMambaAttentionDynamicCache`, not on the `StaticCache` object used by the benchmark harness. This is a model definition / transformers 5.2.0 compatibility issue and cannot be fixed within the benchmark test.

## Decode roofline (first decode graph, single-chip)
Source JSON: N/A (test failed before compilation)
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
- tests/benchmark/test_llms.py (added test_granite_4_0_1b_q4_k_m)
- SUMMARY.md

## tt-forge-models submodule
no change
