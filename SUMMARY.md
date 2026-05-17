loader_path: third_party.tt_forge_models.amd_olmo.causal_lm.pytorch.loader
variant_id: 1b_sft_dpo
arch: n150
status: DONE_PASS
test_function: test_amd_olmo_1b_sft_dpo
samples_per_second: 6.832327252969264
ttft_ms: 857.796226
prefill_pcc: 0.999605
first_decode_pcc: 0.999504
top_perf_samples_per_sec: 128.8277
pct_of_target: 5.3
roofline_bound: dram
optimization_level: 0
trace_enabled: true
experimental_weight_dtype: bfp_bf8
failure_reason: null

# Benchmark added: test_amd_olmo_1b_sft_dpo

## Test
tests/benchmark/test_llms.py::test_amd_olmo_1b_sft_dpo

## Model
- HF name:    amd/AMD-OLMo-1B-SFT-DPO
- Loader:     third_party.tt_forge_models.amd_olmo.causal_lm.pytorch.loader
- Variant:    AMD_OLMo_1B_SFT_DPO ("1b_sft_dpo")

## Test config landed
- optimization_level:        0
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

Note: optimization_level is hard-coded to 0 due to a compiler bug at OL>=1:
'ttnn.scaled_dot_product_attention' op Query and result must have the same element type

## Measured (full model, defaults)
- Sample per second:  6.832327252969264
- TTFT (ms):          857.796226
- Prefill PCC:        0.999605
- First decode PCC:   0.999504
- Wall clock:         0:02:26
- Hardware:           n150 (wormhole_b0, n300 card single-chip)

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_amd_olmo_1b_sft_dpo_perf_metrics_1.json
Achieved vs top_perf_samples_per_sec: 5.3% (6.83 / 128.83)

Note: Low % of target is primarily due to optimization_level=0 being required
(compiler bug prevents OL>=1 from working with this model's SDPA op).

### System
- arch:                        wormhole_b0
- chip_count_in_system_desc:   2
- single_chip_assumption:      True
- worker_grid_cores:           64
- dram_bandwidth_bytes_per_sec: 288000000000

### Peak FLOPs
- lofi:  256000000000000
- hifi2: 128000000000000
- hifi3: 85333333333333
- hifi4: 64000000000000

### Compute
- total_flops:             75849793664
- breakdown.matmul:        75849793664
- breakdown.linear:        0
- breakdown.conv2d:        0
- breakdown.sparse_matmul: 0

### Inputs
- count:        33
- memory_bytes: 132

### KV cache
- count:        268435456
- memory_bytes: 536870912
- memory_gb:    0.5

### Params
- count:                  1279787206
- effective_count:        1176764614
- memory_bytes:           1456358164
- memory_gb:              1.3563392348587513
- effective_memory_bytes: 1250312980
- effective_memory_gb:    1.1644447036087513
- embedding_count:        103022592
- embedding_memory_bytes: 206045184

### Roofline
- bound:                    dram
- top_perf_samples_per_sec: 128.8277
- top_perf_time_ms:         7.7623
- dram_time_ms:             5.1749
- compute_time_ms_lofi:     0.2963
- compute_time_ms_hifi2:    0.5926
- compute_time_ms_hifi3:    0.8889
- compute_time_ms_hifi4:    1.1852

## Files changed
- tests/benchmark/test_llms.py (added test_amd_olmo_1b_sft_dpo)
- tests/benchmark/benchmarks/llm_benchmark.py (infrastructure fix: guard get_weight_dtype_config_path with hasattr)

## tt-forge-models submodule
no change
