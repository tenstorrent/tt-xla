loader_path: third_party.tt_forge_models.helios_nova.causal_lm.pytorch.loader
variant_id: 306M
arch: p150
status: DONE_FAIL
test_function: test_helios_nova_306m
samples_per_second: null
ttft_ms: null
prefill_pcc: null
first_decode_pcc: null
top_perf_samples_per_sec: 1082.2164
pct_of_target: null
roofline_bound: dram
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: bfp_bf8
failure_reason: "Persistent Prefill PCC failure: best PCC=0.377 at opt_level=2 without bfp_bf8 (PCC=0.315 with bfp_bf8), well below required 0.94; opt_level=0 and opt_level=1 crash with ValueError: Error code: 12 (compiler error in TT backend). Infrastructure fix applied: decode_utils.py _ensure_standard_config_attrs() now handles non-standard config attribute names (n_heads/n_kv_heads/n_layers used by HeliosNovaConfig instead of canonical HuggingFace names)."

# Benchmark added: test_helios_nova_306m

## Test
tests/benchmark/test_llms.py::test_helios_nova_306m

## Model
- HF name:    respinosamena/Helios-Nova-306M
- Loader:     third_party.tt_forge_models.helios_nova.causal_lm.pytorch.loader
- Variant:    ModelVariant.HELIOS_NOVA_306M (306M)

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: bfp_bf8
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  N/A (test fails PCC check before recording throughput)
- TTFT (ms):          N/A
- Prefill PCC:        0.377 (best achieved; at opt_level=2 without bfp_bf8; below required 0.94)
- First decode PCC:   N/A (not reached — test aborts at prefill PCC assertion)
- Wall clock:         ~3:00 per attempt
- Hardware:           p150

## Failure analysis
All tested configurations fail:

| Configuration                              | Result                           |
|--------------------------------------------|----------------------------------|
| opt_level=2, bfp_bf8 (default)             | Prefill PCC=0.315 (< 0.94)       |
| opt_level=2, no bfp_bf8                    | Prefill PCC=0.377 (< 0.94)       |
| opt_level=1, no bfp_bf8                    | ValueError: Error code: 12       |
| opt_level=0, no bfp_bf8                    | ValueError: Error code: 12       |

The very low PCC (0.38) even without weight quantization and the compiler crash at
lower optimization levels indicate a fundamental numerical or compiler issue with
HeliosNovaConfig's non-standard model architecture (QK-norm attention, custom
RoPE implementation, non-canonical HF config attributes).

## Infrastructure fix included
`tests/benchmark/llm_utils/decode_utils.py` was updated with two new helpers:
- `_get_config_attr()`: tries multiple candidate attribute names in priority order
- `_ensure_standard_config_attrs()`: patches canonical HuggingFace attribute aliases
  (num_hidden_layers, num_attention_heads, num_key_value_heads) onto configs that use
  alternative names (n_layers, n_heads, n_kv_heads) — such as HeliosNovaConfig.
This is a general fix that benefits any future model with non-standard config attrs.

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_helios_nova_306m_perf_metrics_3.json
Achieved vs top_perf_samples_per_sec: N/A (test failed before stable measurement)

### System
- arch:                        blackhole
- chip_count_in_system_desc:   1
- single_chip_assumption:      True
- worker_grid_cores:           110
- dram_bandwidth_bytes_per_sec: 512000000000

### Peak FLOPs
- lofi:  880000000000000
- hifi2: 440000000000000
- hifi3: 293333333333333
- hifi4: 220000000000000

### Compute
- total_flops:             19573768192
- breakdown.matmul:        19573768192
- breakdown.linear:        0
- breakdown.conv2d:        0
- breakdown.sparse_matmul: 0

### Inputs
- count:        33
- memory_bytes: 132

### KV cache
- count:        0
- memory_bytes: 0
- memory_gb:    0

### Params
- count:                  322228482
- effective_count:        305844482
- memory_bytes:           357777926
- memory_gb:              0.333207
- effective_memory_bytes: 325009926
- effective_memory_gb:    0.302689
- embedding_count:        16384000
- embedding_memory_bytes: 32768000

### Roofline
- bound:                    dram
- top_perf_samples_per_sec: 1082.2164
- top_perf_time_ms:         0.9240
- dram_time_ms:             0.6160
- compute_time_ms_lofi:     0.0222
- compute_time_ms_hifi2:    0.0445
- compute_time_ms_hifi3:    0.0667
- compute_time_ms_hifi4:    0.0890

## Files changed
- tests/benchmark/test_llms.py (added test_helios_nova_306m)
- tests/benchmark/llm_utils/decode_utils.py (added _get_config_attr and _ensure_standard_config_attrs for non-standard config attribute name support)

## tt-forge-models submodule
no change
