loader_path: third_party.tt_forge_models.pytorch_qwen3_4b_int8_int4.causal_lm.pytorch.loader
variant_id: 4B_INT8_INT4
arch: p150
status: DONE_PASS
test_function: test_pytorch_qwen3_4b_int8_int4
samples_per_second: 35.209520271201754
ttft_ms: 326.553405
prefill_pcc: 0.985567
first_decode_pcc: 0.980638
top_perf_samples_per_sec: 550.3531
pct_of_target: 6.4
roofline_bound: compute
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: "bfp_bf8"
failure_reason: null

# Benchmark added: test_pytorch_qwen3_4b_int8_int4

## Test
tests/benchmark/test_llms.py::test_pytorch_qwen3_4b_int8_int4

## Model
- HF name:    pytorch/Qwen3-4B-INT8-INT4
- Loader:     third_party.tt_forge_models.pytorch_qwen3_4b_int8_int4.causal_lm.pytorch.loader
- Variant:    4B_INT8_INT4

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  35.209520271201754
- TTFT (ms):          326.553405
- Prefill PCC:        0.985567
- First decode PCC:   0.980638
- Wall clock:         0:09:17
- Hardware:           p150

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_pytorch_qwen3_4b_int8_int4_perf_metrics_0.json
Achieved vs top_perf_samples_per_sec: 6.4%

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
- total_flops:             532991182976
- breakdown.matmul:        532991182976
- breakdown.linear:        0
- breakdown.conv2d:        0
- breakdown.sparse_matmul: 0

### Inputs
- count:        561
- memory_bytes: 2244

### KV cache
- count:        8388608
- memory_bytes: 16777216
- memory_gb:    0.015625

### Params
- count:                  878845891
- effective_count:        489889731
- memory_bytes:           1298428168
- memory_gb:              1.209255464375019
- effective_memory_bytes: 520515848
- effective_memory_gb:    0.4847681596875191
- embedding_count:        388956160
- embedding_memory_bytes: 777912320

### Roofline
- bound:                    compute
- top_perf_samples_per_sec: 550.3531
- top_perf_time_ms:         1.8170
- dram_time_ms:             1.0036
- compute_time_ms_lofi:     0.6057
- compute_time_ms_hifi2:    1.2113
- compute_time_ms_hifi3:    1.8170
- compute_time_ms_hifi4:    2.4227

## Files changed
- tests/benchmark/test_llms.py (test function already present)
- tests/benchmark/benchmarks/llm_benchmark.py (defensive getattr for get_weight_dtype_config_path)

## tt-forge-models submodule
no change — submodule remains at f7aabc34a7 (ip-172-31-30-236-tt-xla-dev/ubuntu/2026-04-23_02-28/hf-bringup-36 branch)
