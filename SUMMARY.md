loader_path: third_party.tt_forge_models.captainerisnebula_12b_aoe_v1_i1_gguf.causal_lm.pytorch.loader
variant_id: CAPTAINERISNEBULA_12B_AOE_V1_I1_Q4_K_M_GGUF
arch: p150
status: DONE_PASS
test_function: test_captainerisnebula_12b_aoe_v1_i1_gguf
samples_per_second: 55.47
ttft_ms: 73.75
prefill_pcc: 0.998902
first_decode_pcc: 0.999134
top_perf_samples_per_sec: 268.41
pct_of_target: 20.7
roofline_bound: dram
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: bfp_bf8
failure_reason: null

# Benchmark added: test_captainerisnebula_12b_aoe_v1_i1_gguf

## Test
tests/benchmark/test_llms.py::test_captainerisnebula_12b_aoe_v1_i1_gguf

## Model
- HF name:    mradermacher/CaptainErisNebula-12B-AOE-v1-i1-GGUF
- Loader:     third_party.tt_forge_models.captainerisnebula_12b_aoe_v1_i1_gguf.causal_lm.pytorch.loader
- Variant:    CAPTAINERISNEBULA_12B_AOE_V1_I1_Q4_K_M_GGUF

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: bfp_bf8
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

Note: The loader sets DEFAULT_NUM_LAYERS=2 to avoid DRAM OOM on a single device.
When num_layers is None (pytest default), the model runs with 2 layers. The
benchmark metrics below reflect the 2-layer configuration.

## Measured (full model, defaults)
- Sample per second:  55.47
- TTFT (ms):          73.75
- Prefill PCC:        0.998902
- First decode PCC:   0.999134
- Wall clock:         0:06:08
- Hardware:           p150

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_captainerisnebula_12b_aoe_v1_i1_gguf_perf_metrics_1.json
Achieved vs top_perf_samples_per_sec: 20.7% (55.47 / 268.41)

### System
- arch:                        blackhole
- chip_count_in_system_desc:   1
- single_chip_assumption:      True
- worker_grid_cores:           130
- dram_bandwidth_bytes_per_sec: 512000000000

### Peak FLOPs
- lofi:  1040000000000000
- hifi2: 520000000000000
- hifi3: 346666666666666
- hifi4: 260000000000000

### Compute
- total_flops:             77980500096
- breakdown.matmul:        77980500096
- breakdown.linear:        0
- breakdown.conv2d:        0
- breakdown.sparse_matmul: 0

### Inputs
- count:        33
- memory_bytes: 132

### KV cache
- count:        16777216
- memory_bytes: 33554432
- memory_gb:    0.03125

### Params
- count:                  1887462598
- effective_count:        1216373958
- memory_bytes:           2634599188
- memory_gb:              2.453661698848009
- effective_memory_bytes: 1292421908
- effective_memory_gb:    1.203661698848009
- embedding_count:        671088640
- embedding_memory_bytes: 1342177280

### Roofline
- bound:                    dram
- top_perf_samples_per_sec: 268.4098
- top_perf_time_ms:         3.7256
- dram_time_ms:             2.4838
- compute_time_ms_lofi:     0.0750
- compute_time_ms_hifi2:    0.1500
- compute_time_ms_hifi3:    0.2249
- compute_time_ms_hifi4:    0.2999

## Files changed
- tests/benchmark/test_llms.py (added test_captainerisnebula_12b_aoe_v1_i1_gguf)
- tests/benchmark/benchmarks/llm_benchmark.py (general fix: guard get_weight_dtype_config_path with hasattr)

## tt-forge-models submodule
no change
