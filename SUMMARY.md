loader_path: third_party.tt_forge_models.daixuancheng_qwen3_4b_instruct_2507_llm_in_sandbox_rl.causal_lm.pytorch.loader
variant_id: 4B_Instruct_2507_LLM_in_Sandbox_RL
arch: n150
status: DONE_PASS
test_function: test_daixuancheng_qwen3_4b_instruct_2507_llm_in_sandbox_rl
samples_per_second: 17.411853792896704
ttft_ms: 692.298089
prefill_pcc: 0.996560
first_decode_pcc: 0.998505
top_perf_samples_per_sec: 43.0532
pct_of_target: 40.4
roofline_bound: dram
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: bfp_bf8
failure_reason: null

# Benchmark added: test_daixuancheng_qwen3_4b_instruct_2507_llm_in_sandbox_rl

## Test
tests/benchmark/test_llms.py::test_daixuancheng_qwen3_4b_instruct_2507_llm_in_sandbox_rl

## Model
- HF name:    daixuancheng/Qwen3-4B-Instruct-2507-LLM-in-Sandbox-RL
- Loader:     third_party.tt_forge_models.daixuancheng_qwen3_4b_instruct_2507_llm_in_sandbox_rl.causal_lm.pytorch.loader
- Variant:    4B_Instruct_2507_LLM_in_Sandbox_RL

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  17.41
- TTFT (ms):          692.30
- Prefill PCC:        0.996560
- First decode PCC:   0.998505
- Wall clock:         0:14:38
- Hardware:           n150

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_daixuancheng_qwen3_4b_instruct_2507_llm_in_sandbox_rl_perf_metrics_3.json
Achieved vs top_perf_samples_per_sec: 40.4%

### System
- arch:                        wormhole_b0
- chip_count_in_system_desc:   2
- single_chip_assumption:      True
- worker_grid_cores:           64
- dram_bandwidth_bytes_per_sec: 288000000000

### Peak FLOPs
- lofi:  256000000000000
- hifi2: 128000000000000
- hifi3: 85333333333333
- hifi4: 64000000000000

### Compute
- total_flops:             257425408128
- breakdown.matmul:        257425408128
- breakdown.linear:        0
- breakdown.conv2d:        0
- breakdown.sparse_matmul: 0

### Inputs
- count:        33
- memory_bytes: 132

### KV cache
- count:        301989888
- memory_bytes: 603979776
- memory_gb:    0.5625

### Params
- count:                  4411424451
- effective_count:        4022468291
- memory_bytes:           5051969288
- memory_gb:              4.705013044178486
- effective_memory_bytes: 4274056968
- effective_memory_gb:    3.980525739490986
- embedding_count:        388956160
- embedding_memory_bytes: 777912320

### Roofline
- bound:                    dram
- top_perf_samples_per_sec: 43.0532
- top_perf_time_ms:         23.2271
- dram_time_ms:             15.4847
- compute_time_ms_lofi:     1.0056
- compute_time_ms_hifi2:    2.0111
- compute_time_ms_hifi3:    3.0167
- compute_time_ms_hifi4:    4.0223

## Files changed
- tests/benchmark/test_llms.py
- tests/benchmark/benchmarks/llm_benchmark.py
- .github/workflows/perf-bench-matrix.json

## tt-forge-models submodule
no change
