loader_path: third_party.tt_forge_models.qwen_3_0_6b_code_expert.causal_lm.pytorch.loader
variant_id: 0_6B_Code_Expert
arch: p150
status: DONE_PASS
test_function: test_qwen_3_0_6b_code_expert
samples_per_second: 75.24
ttft_ms: 203.87
prefill_pcc: 0.991270
first_decode_pcc: 0.998182
top_perf_samples_per_sec: 398.3361
pct_of_target: 18.9
roofline_bound: dram
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: bfp_bf8
failure_reason: null

# Benchmark added: test_qwen_3_0_6b_code_expert

## Test
tests/benchmark/test_llms.py::test_qwen_3_0_6b_code_expert

## Model
- HF name:    suayptalha/Qwen3-0.6B-Code-Expert
- Loader:     third_party.tt_forge_models.qwen_3_0_6b_code_expert.causal_lm.pytorch.loader
- Variant:    ModelVariant.QWEN_3_0_6B_CODE_EXPERT

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  75.24
- TTFT (ms):          203.87
- Prefill PCC:        0.991270
- First decode PCC:   0.998182
- Wall clock:         0:04:49
- Hardware:           p150

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_qwen_3_0_6b_code_expert_perf_metrics_1.json
Achieved vs top_perf_samples_per_sec: 75.24 / 398.34 = 18.9%

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
- total_flops:             38143000704
- breakdown.matmul:        38143000704
- breakdown.linear:        0
- breakdown.conv2d:        0
- breakdown.sparse_matmul: 0

### Inputs
- count:        33
- memory_bytes: 132

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
- compute_time_ms_lofi:     0.0433
- compute_time_ms_hifi2:    0.0867
- compute_time_ms_hifi3:    0.1300
- compute_time_ms_hifi4:    0.1734

## Files changed
- tests/benchmark/test_llms.py (new test: test_qwen_3_0_6b_code_expert)
- tests/benchmark/benchmarks/llm_benchmark.py (infrastructure fix: guard get_weight_dtype_config_path with hasattr)

## tt-forge-models submodule
no change (submodule at 70f1328366)
