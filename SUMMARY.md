loader_path: third_party.tt_forge_models.amd_olmo.causal_lm.pytorch.loader
variant_id: 1b_sft_dpo
arch: p150
status: DONE_PASS
test_function: test_amd_olmo_1b_sft_dpo
samples_per_second: 13.45314169890514
ttft_ms: 392.644579
prefill_pcc: 0.998823
first_decode_pcc: 0.999064
top_perf_samples_per_sec: 229.0270
pct_of_target: 5.9
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

Note: optimization_level=0 is required — OL>=1 triggers compiler error:
'ttnn.scaled_dot_product_attention' op Query and result must have the same
element type (confirmed on both n150 and p150 at this submodule HEAD).

## Measured (full model, defaults)
- Sample per second:  13.45314169890514
- TTFT (ms):          392.644579
- Prefill PCC:        0.998823
- First decode PCC:   0.999064
- Wall clock:         0:01:57
- Hardware:           p150

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_amd_olmo_1b_sft_dpo_perf_metrics_1.json
Achieved vs top_perf_samples_per_sec: 5.9% (13.45 / 229.03)

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
- top_perf_samples_per_sec: 229.0270
- top_perf_time_ms:         4.3663
- dram_time_ms:             2.9109
- compute_time_ms_lofi:     0.0862
- compute_time_ms_hifi2:    0.1724
- compute_time_ms_hifi3:    0.2586
- compute_time_ms_hifi4:    0.3448

## Files changed
- tests/benchmark/test_llms.py
- .github/workflows/perf-bench-matrix.json

## tt-forge-models submodule
no change
