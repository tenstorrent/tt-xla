loader_path: third_party.tt_forge_models.nanbeige_4_1_3b_q8_0_gguf.causal_lm.pytorch.loader
variant_id: 3B_Q8_0_GGUF
arch: p150
status: DONE_PASS
test_function: test_nanbeige_4_1_3b_q8_0_gguf
samples_per_second: 60.255
ttft_ms: 205.930
prefill_pcc: 0.996280
first_decode_pcc: 0.975106
top_perf_samples_per_sec: 90.8682
pct_of_target: 66.3
roofline_bound: dram
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: "bfp_bf8"
failure_reason: null

# Benchmark added: nanbeige_4_1_3b_q8_0_gguf

## Test
tests/benchmark/test_llms.py::test_nanbeige_4_1_3b_q8_0_gguf

## Model
- HF name:    Edge-Quant/Nanbeige4.1-3B-Q8_0-GGUF
- Loader:     third_party.tt_forge_models.nanbeige_4_1_3b_q8_0_gguf.causal_lm.pytorch.loader
- Variant:    ModelVariant.NANBEIGE_4_1_3B_Q8_0_GGUF

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  60.255
- TTFT (ms):          205.930
- Prefill PCC:        0.996280
- First decode PCC:   0.975106
- Wall clock:         0:14:06
- Hardware:           p150

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_nanbeige_4_1_3b_q8_0_gguf_perf_metrics_1.json
Achieved vs top_perf_samples_per_sec: 66.3%

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
- total_flops:             224521093248
- breakdown.matmul:        224521093248
- breakdown.linear:        0
- breakdown.conv2d:        0
- breakdown.sparse_matmul: 0

### Inputs
- count:        33
- memory_bytes: 132

### KV cache
- count:        134217728
- memory_bytes: 268435456
- memory_gb:    0.25

### Params
- count:                  3933637315
- effective_count:        3508308675
- memory_bytes:           4578391816
- memory_gb:              4.263959653675556
- effective_memory_bytes: 3727734536
- effective_memory_gb:    3.471723325550556
- embedding_count:        425328640
- embedding_memory_bytes: 850657280

### Roofline
- bound:                    dram
- top_perf_samples_per_sec: 90.8682
- top_perf_time_ms:         11.0049
- dram_time_ms:             7.3366
- compute_time_ms_lofi:     0.2551
- compute_time_ms_hifi2:    0.5103
- compute_time_ms_hifi3:    0.7654
- compute_time_ms_hifi4:    1.0206

## Files changed
- tests/benchmark/test_llms.py
- tests/benchmark/benchmarks/llm_benchmark.py

## tt-forge-models submodule
no change
