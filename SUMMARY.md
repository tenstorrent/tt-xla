loader_path: third_party.tt_forge_models.koboldai_gpt_j_6b_skein.causal_lm.pytorch.loader
variant_id: default
arch: p150
status: DONE_PASS
test_function: test_koboldai_gpt_j_6b_skein
samples_per_second: 7.400155577359148
ttft_ms: 1244.549066
prefill_pcc: 0.987557
first_decode_pcc: 0.998432
top_perf_samples_per_sec: 48.763630060933714
pct_of_target: 15.2
roofline_bound: dram
optimization_level: 1
trace_enabled: true
experimental_weight_dtype: bfp_bf8
failure_reason: null

# Benchmark added: test_koboldai_gpt_j_6b_skein

## Test
tests/benchmark/test_llms.py::test_koboldai_gpt_j_6b_skein

## Model
- HF name:    KoboldAI/GPT-J-6B-Skein
- Loader:     third_party.tt_forge_models.koboldai_gpt_j_6b_skein.causal_lm.pytorch.loader
- Variant:    ModelVariant.DEFAULT

## Test config landed
- optimization_level:        1
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  7.400155577359148
- TTFT (ms):          1244.549066
- Prefill PCC:        0.987557
- First decode PCC:   0.998432
- Wall clock:         0:03:57
- Hardware:           p150

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_koboldai_gpt_j_6b_skein_perf_metrics_2.json
Achieved vs top_perf_samples_per_sec: 15.2% (7.40 / 48.76)

Note: optimization_level=2 fails with "Physical shard shape (17, 32) must be
tile {32, 32} sized!" — compiler shard alignment issue specific to GPT-J
architecture. Using optimization_level=1 as the best stable config.

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
- total_flops:             374009273344
- breakdown.matmul:        120259084288
- breakdown.linear:        253750189056
- breakdown.conv2d:        0
- breakdown.sparse_matmul: 0

### Inputs
- count:        33
- memory_bytes: 132

### KV cache
- count:        939524096
- memory_bytes: 1879048192
- memory_gb:    1.75

### Params
- count:                  6054552931
- effective_count:        5848114531
- memory_bytes:           6630747080
- memory_gb:              6.175364442169666
- effective_memory_bytes: 6217870280
- effective_memory_gb:    5.790842957794666
- embedding_count:        206438400
- embedding_memory_bytes: 412876800

### Roofline
- bound:                    dram
- top_perf_samples_per_sec: 48.763630060933714
- top_perf_time_ms:         20.5071
- dram_time_ms:             13.6714
- compute_time_ms_lofi:     0.4250
- compute_time_ms_hifi2:    0.8500
- compute_time_ms_hifi3:    1.2750
- compute_time_ms_hifi4:    1.7000

## Files changed
- tests/benchmark/test_llms.py (added test_koboldai_gpt_j_6b_skein)
- tests/benchmark/benchmarks/llm_benchmark.py (general fix: guard get_weight_dtype_config_path with hasattr)

## tt-forge-models submodule
no change
