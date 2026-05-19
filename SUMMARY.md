loader_path: third_party.tt_forge_models.deepseek_r1_distill_llama_8b_4bit.causal_lm.pytorch.loader
variant_id: DeepSeek_R1_Distill_Llama_8B_4bit
arch: p150
status: DONE_PASS
test_function: test_deepseek_r1_distill_llama_8b_4bit
samples_per_second: 4.596345565314676
ttft_ms: 808.893583
prefill_pcc: 0.995775
first_decode_pcc: 0.995577
top_perf_samples_per_sec: 42.5800
pct_of_target: 10.8
roofline_bound: dram
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: bfp_bf8
failure_reason: null

# Benchmark added: test_deepseek_r1_distill_llama_8b_4bit

## Test
tests/benchmark/test_llms.py::test_deepseek_r1_distill_llama_8b_4bit

## Model
- HF name:    mlx-community/DeepSeek-R1-Distill-Llama-8B-4bit
- Loader:     third_party.tt_forge_models.deepseek_r1_distill_llama_8b_4bit.causal_lm.pytorch.loader
- Variant:    DeepSeek_R1_Distill_Llama_8B_4bit

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  4.596345565314676
- TTFT (ms):          808.893583
- Prefill PCC:        0.995775
- First decode PCC:   0.995577
- Wall clock:         0:47:08
- Hardware:           p150

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_deepseek_r1_distill_llama_8b_4bit_perf_metrics_1.json
Achieved vs top_perf_samples_per_sec: 10.8% (4.60 / 42.58)

### System
- arch:                        blackhole
- chip_count_in_system_desc:   1
- single_chip_assumption:      True
- worker_grid_cores:           130
- dram_bandwidth_bytes_per_sec: 512000000000

### Peak FLOPs
- lofi:  1040000000000000
- hifi2: 520000000000000
- hifi3: 346666666666666
- hifi4: 260000000000000

### Compute
- total_flops:             482445623424
- breakdown.matmul:        482445623424
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
- count:                  8030261446
- effective_count:        7504924870
- memory_bytes:           9024906004
- memory_gb:              8.40509869530797
- effective_memory_bytes: 7974232852
- effective_memory_gb:    7.42658307030797
- embedding_count:        525336576
- embedding_memory_bytes: 1050673152

### Roofline
- bound:                    dram
- top_perf_samples_per_sec: 42.5800
- top_perf_time_ms:         23.4852
- dram_time_ms:             15.6568
- compute_time_ms_lofi:     0.4639
- compute_time_ms_hifi2:    0.9278
- compute_time_ms_hifi3:    1.3917
- compute_time_ms_hifi4:    1.8556

## Files changed
- tests/benchmark/test_llms.py
- tests/benchmark/benchmarks/llm_benchmark.py

## tt-forge-models submodule
no change
