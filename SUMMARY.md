loader_path: third_party.tt_forge_models.llama_swallow.causal_lm.pytorch.loader
variant_id: 3.1_Swallow_8B_v0.5
arch: p150
status: DONE_PASS
test_function: test_llama_swallow_3_1_8b
samples_per_second: 33.564017387848445
ttft_ms: 311.379061
prefill_pcc: 0.997658
first_decode_pcc: 0.998195
top_perf_samples_per_sec: 33.92954281250499
pct_of_target: 98.9
roofline_bound: compute
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: "bfp_bf8"
failure_reason: null

# Benchmark added: test_llama_swallow_3_1_8b

## Test
tests/benchmark/test_llms.py::test_llama_swallow_3_1_8b

## Model
- HF name:    tokyotech-llm/Llama-3.1-Swallow-8B-v0.5
- Loader:     third_party.tt_forge_models.llama_swallow.causal_lm.pytorch.loader
- Variant:    ModelVariant.LLAMA_3_1_SWALLOW_8B_V0_5

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  33.564017387848445
- TTFT (ms):          311.379061
- Prefill PCC:        0.997658
- First decode PCC:   0.998195
- Wall clock:         0:19:04
- Hardware:           p150

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_llama_swallow_3_1_8b_perf_metrics_1.json
Achieved vs top_perf_samples_per_sec: 98.9%

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
- total_flops:             8645366515968
- breakdown.matmul:        8645366515968
- breakdown.linear:        0
- breakdown.conv2d:        0
- breakdown.sparse_matmul: 0

### Inputs
- count:        594
- memory_bytes: 2376

### KV cache
- count:        268435456
- memory_bytes: 536870912
- memory_gb:    0.5

### Params
- count:                  8030261443
- effective_count:        7504924867
- memory_bytes:           9024905992
- memory_gb:              8.4050986841321
- effective_memory_bytes: 7974232840
- effective_memory_gb:    7.426583059132099
- embedding_count:        525336576
- embedding_memory_bytes: 1050673152

### Roofline
- bound:                    compute
- top_perf_samples_per_sec: 33.92954281250499
- top_perf_time_ms:         29.47
- dram_time_ms:             15.66
- compute_time_ms_lofi:     9.82
- compute_time_ms_hifi2:    19.65
- compute_time_ms_hifi3:    29.47
- compute_time_ms_hifi4:    39.30

## Files changed
- tests/benchmark/test_llms.py

## tt-forge-models submodule
no change
