loader_path: third_party.tt_forge_models.qwen_3_0_6b_gensyn_swarm_finicky_bristly_lion.causal_lm.pytorch.loader
variant_id: finicky_bristly_lion
arch: p150
status: DONE_PASS
test_function: test_qwen_3_0_6b_gensyn_swarm_finicky_bristly_lion
samples_per_second: 76.41954209395094
ttft_ms: 196.981859
prefill_pcc: 0.993209
first_decode_pcc: 0.997739
top_perf_samples_per_sec: 398.3361
pct_of_target: 19.2
roofline_bound: dram
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: bfp_bf8
failure_reason: null

# Benchmark added: test_qwen_3_0_6b_gensyn_swarm_finicky_bristly_lion

## Test
tests/benchmark/test_llms.py::test_qwen_3_0_6b_gensyn_swarm_finicky_bristly_lion

## Model
- HF name:    nick00991/Qwen3-0.6B-Gensyn-Swarm-finicky_bristly_lion
- Loader:     third_party.tt_forge_models.qwen_3_0_6b_gensyn_swarm_finicky_bristly_lion.causal_lm.pytorch.loader
- Variant:    ModelVariant.FINICKY_BRISTLY_LION

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  76.41954209395094
- TTFT (ms):          196.981859
- Prefill PCC:        0.993209
- First decode PCC:   0.997739
- Wall clock:         0:04:33
- Hardware:           p150

## Decode roofline (first decode graph, single-chip)
Source JSON: tests/benchmark/tt_xla_qwen_3_0_6b_gensyn_swarm_finicky_bristly_lion_perf_metrics_0.json
Achieved vs top_perf_samples_per_sec: 19.2%

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
- total_flops:             648431011968
- breakdown.matmul:        648431011968
- breakdown.linear:        0
- breakdown.conv2d:        0
- breakdown.sparse_matmul: 0

### Inputs
- count:        561
- memory_bytes: 2244

### KV cache
- count:        234881024
- memory_bytes: 469762048
- memory_gb:    0.4375

### Params
- count:                  751632579
- effective_count:        596050115
- memory_bytes:           944530184
- memory_gb:              0.8796622827649117
- effective_memory_bytes: 633365256
- effective_memory_gb:    0.5898673608899117
- embedding_count:        155582464
- embedding_memory_bytes: 311164928

### Roofline
- bound:                    dram
- top_perf_samples_per_sec: 398.3361
- top_perf_time_ms:         2.5104
- dram_time_ms:             1.6736
- compute_time_ms_lofi:     0.7369
- compute_time_ms_hifi2:    1.4737
- compute_time_ms_hifi3:    2.2106
- compute_time_ms_hifi4:    2.9474

## Files changed
- tests/benchmark/test_llms.py
- tests/benchmark/benchmarks/llm_benchmark.py

## tt-forge-models submodule
no change
