loader_path: third_party.tt_forge_models.kanana_nano_2_1b_instruct.causal_lm.pytorch.loader
variant_id: Kanana_Nano_2_1B_Instruct
arch: p150
status: DONE_PASS
test_function: test_kanana_nano_2_1b_instruct
samples_per_second: 63.07
ttft_ms: 220.51
prefill_pcc: 0.997177
first_decode_pcc: 0.997803
top_perf_samples_per_sec: 140.5230
pct_of_target: 44.9
roofline_bound: dram
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: bfp_bf8
failure_reason: null

# Benchmark added: test_kanana_nano_2_1b_instruct

## Test
tests/benchmark/test_llms.py::test_kanana_nano_2_1b_instruct

## Model
- HF name:    kakaocorp/kanana-nano-2.1b-instruct
- Loader:     third_party.tt_forge_models.kanana_nano_2_1b_instruct.causal_lm.pytorch.loader
- Variant:    Kanana_Nano_2_1B_Instruct

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  63.07
- TTFT (ms):          220.51
- Prefill PCC:        0.997177
- First decode PCC:   0.997803
- Wall clock:         0:05:50
- Hardware:           p150

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_kanana_nano_2_1b_instruct_perf_metrics_1.json
Achieved vs top_perf_samples_per_sec: 44.9%

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
- total_flops:             133559222400
- breakdown.matmul:        133559222400
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
- count:                  2316814275
- effective_count:        2086979523
- memory_bytes:           2677195016
- memory_gb:              2.493332155048847
- effective_memory_bytes: 2217525512
- effective_memory_gb:    2.065231569111347
- embedding_count:        229834752
- embedding_memory_bytes: 459669504

### Roofline
- bound:                    dram
- top_perf_samples_per_sec: 140.5230
- top_perf_time_ms:         7.1163
- dram_time_ms:             4.7442
- compute_time_ms_lofi:     0.1518
- compute_time_ms_hifi2:    0.3035
- compute_time_ms_hifi3:    0.4553
- compute_time_ms_hifi4:    0.6071

## Files changed
- tests/benchmark/test_llms.py (added test_kanana_nano_2_1b_instruct)
- tests/benchmark/benchmarks/llm_benchmark.py (general fix: use hasattr before calling get_weight_dtype_config_path)

## tt-forge-models submodule
no change
