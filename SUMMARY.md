loader_path: third_party.tt_forge_models.redpajama_incite_chat_3b_v1_q4f16_1_mlc.causal_lm.pytorch.loader
variant_id: RedPajama_INCITE_Chat_3B_v1_Q4F16_1_MLC
arch: p150
status: DONE_PASS
test_function: test_redpajama_incite_chat_3b_v1_q4f16_1_mlc
samples_per_second: 21.687546177591575
ttft_ms: 565.923672
prefill_pcc: 0.984264
first_decode_pcc: 0.994274
top_perf_samples_per_sec: 99.7604
pct_of_target: 21.7
roofline_bound: dram
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: bfp_bf8
failure_reason: null

# Benchmark added: test_redpajama_incite_chat_3b_v1_q4f16_1_mlc

## Test
tests/benchmark/test_llms.py::test_redpajama_incite_chat_3b_v1_q4f16_1_mlc

## Model
- HF name:    togethercomputer/RedPajama-INCITE-Chat-3B-v1
- Loader:     third_party.tt_forge_models.redpajama_incite_chat_3b_v1_q4f16_1_mlc.causal_lm.pytorch.loader
- Variant:    RedPajama_INCITE_Chat_3B_v1_Q4F16_1_MLC

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: bfp_bf8
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  21.687546177591575
- TTFT (ms):          565.923672
- Prefill PCC:        0.984264
- First decode PCC:   0.994274
- Wall clock:         0:06:28
- Hardware:           p150

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_redpajama_incite_chat_3b_v1_q4f16_1_mlc_perf_metrics_1.json
Achieved vs top_perf_samples_per_sec: 21.7% (21.69 / 99.76)

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
- total_flops:             169347645520
- breakdown.matmul:        8262778960
- breakdown.linear:        161084866560
- breakdown.conv2d:        0
- breakdown.sparse_matmul: 0

### Inputs
- count:        33
- memory_bytes: 132

### KV cache
- count:        671088640
- memory_bytes: 1342177280
- memory_gb:    1.25

### Params
- count:                  2775864491
- effective_count:        2646758571
- memory_bytes:           3071396520
- memory_gb:              2.860460914671421
- effective_memory_bytes: 2813184680
- effective_memory_gb:    2.619982399046421
- embedding_count:        129105920
- embedding_memory_bytes: 258211840

### Roofline
- bound:                    dram
- top_perf_samples_per_sec: 99.7604
- top_perf_time_ms:         10.0240
- dram_time_ms:             6.6827
- compute_time_ms_lofi:     0.1924
- compute_time_ms_hifi2:    0.3849
- compute_time_ms_hifi3:    0.5773
- compute_time_ms_hifi4:    0.7698

## Files changed
- tests/benchmark/test_llms.py (new test function added)
- tests/benchmark/benchmarks/llm_benchmark.py (fix: guard get_weight_dtype_config_path with hasattr)

## tt-forge-models submodule
no change (submodule at ae1ffad0da)
