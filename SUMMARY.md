loader_path: third_party.tt_forge_models.jais_2_8b_chat.causal_lm.pytorch.loader
variant_id: jais_2_8b_chat
arch: n150
status: DONE_PASS
test_function: test_jais_2_8b_chat
samples_per_second: 21.8
ttft_ms: 462.9
prefill_pcc: 0.992799
first_decode_pcc: 0.996349
top_perf_samples_per_sec: 39.1116
pct_of_target: 55.7
roofline_bound: dram
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: bfp_bf8
failure_reason: null

# Benchmark added: test_jais_2_8b_chat

## Test
tests/benchmark/test_llms.py::test_jais_2_8b_chat

## Model
- HF name:    Omaratef3221/jais-2-8b-chat-s1-full-aramed
- Loader:     third_party.tt_forge_models.jais_2_8b_chat.causal_lm.pytorch.loader
- Variant:    ModelVariant.JAIS_2_8B_CHAT

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  21.8
- TTFT (ms):          462.9
- Prefill PCC:        0.992799
- First decode PCC:   0.996349
- Wall clock:         0:09:33
- Hardware:           n150

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_jais_2_8b_chat_perf_metrics_1.json
Achieved vs top_perf_samples_per_sec: 55.7%

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
- total_flops:             485706956928
- breakdown.matmul:        32006733952
- breakdown.linear:        453700222976
- breakdown.conv2d:        0
- breakdown.sparse_matmul: 0

### Inputs
- count:        33
- memory_bytes: 132

### KV cache
- count:        872415232
- memory_bytes: 1744830464
- memory_gb:    1.625

### Params
- count:                  8090401475
- effective_count:        7590296259
- memory_bytes:           9066604296
- memory_gb:              8.443933255970478
- effective_memory_bytes: 8066393864
- effective_memory_gb:    7.512414701282978
- embedding_count:        500105216
- embedding_memory_bytes: 1000210432

### Roofline
- bound:                    dram
- top_perf_samples_per_sec: 39.1116
- top_perf_time_ms:         25.5679
- dram_time_ms:             17.0453
- compute_time_ms_lofi:     0.5519
- compute_time_ms_hifi2:    1.1039
- compute_time_ms_hifi3:    1.6558
- compute_time_ms_hifi4:    2.2078

## Files changed
- tests/benchmark/test_llms.py
- tests/benchmark/benchmarks/llm_benchmark.py (infrastructure fix: defensive getattr for get_weight_dtype_config_path)

## tt-forge-models submodule
no change
