loader_path: third_party.tt_forge_models.mythomax_l2_kimiko.causal_lm.pytorch.loader
variant_id: MythoMax_L2_Kimiko_v2_13B
arch: p150
status: DONE_FAIL
test_function: test_mythomax_l2_kimiko_v2_13b
samples_per_second: 13.935107695675933
ttft_ms: 567.544233
prefill_pcc: 0.999294
first_decode_pcc: 0.895796
top_perf_samples_per_sec: 22.7802
pct_of_target: 61.2
roofline_bound: dram
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: "bfp_bf8"
failure_reason: "First decode PCC=0.895796 below required=0.94 at optimization_level=2; also tried optimization_level=1 (PCC=0.626067) and no weight quantization (PCC=0.891006) — numerical accuracy accumulation issue in decode path across 40 layers"

# Benchmark added: test_mythomax_l2_kimiko_v2_13b

## Test
tests/benchmark/test_llms.py::test_mythomax_l2_kimiko_v2_13b

## Model
- HF name:    Undi95/MythoMax-L2-Kimiko-v2-13b
- Loader:     third_party.tt_forge_models.mythomax_l2_kimiko.causal_lm.pytorch.loader
- Variant:    MythoMax_L2_Kimiko_v2_13B

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  13.935107695675933
- TTFT (ms):          567.544233
- Prefill PCC:        0.999294
- First decode PCC:   0.895796 (FAIL — below required 0.94)
- Wall clock:         0:12:49
- Hardware:           p150

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_mythomax_l2_kimiko_v2_13b_perf_metrics_3.json
Achieved vs top_perf_samples_per_sec: 61.2%

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
- total_flops:             822503014528
- breakdown.matmul:        822503014528
- breakdown.linear:        0
- breakdown.conv2d:        0
- breakdown.sparse_matmul: 0

### Inputs
- count:        33
- memory_bytes: 132

### KV cache
- count:        1677721600
- memory_bytes: 3355443200
- memory_gb:    3.125

### Params
- count:                  13015864515
- effective_count:        12852024515
- memory_bytes:           26031729416
- memory_gb:              24.243937261402607
- effective_memory_bytes: 25704049416
- effective_memory_gb:    23.938761480152607
- embedding_count:        163840000
- embedding_memory_bytes: 327680000

### Roofline
- bound:                    dram
- top_perf_samples_per_sec: 22.7802
- top_perf_time_ms:         43.8979
- dram_time_ms:             29.2652
- compute_time_ms_lofi:     0.9347
- compute_time_ms_hifi2:    1.8693
- compute_time_ms_hifi3:    2.8040
- compute_time_ms_hifi4:    3.7387

## Files changed
- tests/benchmark/test_llms.py
- tests/benchmark/benchmarks/llm_benchmark.py (general fix: gracefully handle loaders without get_weight_dtype_config_path)

## tt-forge-models submodule
no change
