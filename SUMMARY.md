loader_path: third_party.tt_forge_models.sip_jmed_llm_3_8x13b_op_32k_r0_1_gguf.causal_lm.pytorch.loader
variant_id: 3_8x13b_OP_32k_R0.1_GGUF
arch: p150
status: DONE_PASS
test_function: test_sip_jmed_llm_3_8x13b_op_32k_r0_1_gguf
samples_per_second: 12.35
ttft_ms: 662.93
prefill_pcc: 0.999148
first_decode_pcc: 0.994686
top_perf_samples_per_sec: 22.2503
pct_of_target: 55.5
roofline_bound: dram
optimization_level: 1
trace_enabled: true
experimental_weight_dtype: bfp_bf8
failure_reason: null

# Benchmark added: test_sip_jmed_llm_3_8x13b_op_32k_r0_1_gguf

## Test
tests/benchmark/test_llms.py::test_sip_jmed_llm_3_8x13b_op_32k_r0_1_gguf

## Model
- HF name:    hiratagoh/SIP-jmed-llm-3-8x13b-OP-32k-R0.1-GGUF
- Loader:     third_party.tt_forge_models.sip_jmed_llm_3_8x13b_op_32k_r0_1_gguf.causal_lm.pytorch.loader
- Variant:    ModelVariant.SIP_JMED_LLM_3_8X13B_OP_32K_R0_1_GGUF

## Test config landed
- optimization_level:        1
- trace_enabled:             true
- experimental_weight_dtype: bfp_bf8
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

Note: optimization_level=2 fails first decode PCC (0.584311 < 0.94); optimization_level=1 passes (0.994686).
Also fixed a general harness bug: benchmarks/llm_benchmark.py called model_loader.get_weight_dtype_config_path()
unconditionally, but the ForgeModel base class does not define this method. Fixed with hasattr() guard.

## Measured (full model, defaults)
- Sample per second:  12.35
- TTFT (ms):          662.93
- Prefill PCC:        0.999148
- First decode PCC:   0.994686
- Wall clock:         0:12:53
- Hardware:           p150

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_sip_jmed_llm_3_8x13b_op_32k_r0_1_gguf_perf_metrics_1.json
Achieved vs top_perf_samples_per_sec: 55.5% (12.35 / 22.25)

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
- total_flops:             844648939648
- breakdown.matmul:        844648939648
- breakdown.linear:        0
- breakdown.conv2d:        0
- breakdown.sparse_matmul: 0

### Inputs
- count:        33
- memory_bytes: 132

### KV cache
- count:        1677721600
- memory_bytes: 3355443200
- memory_gb:    3.125

### Params
- count:                  13707924675
- effective_count:        13198054595
- memory_bytes:           15043062536
- memory_gb:              14.00994373112917
- effective_memory_bytes: 14023322376
- effective_memory_gb:    13.06023669987917
- embedding_count:        509870080
- embedding_memory_bytes: 1019740160

### Roofline
- bound:                    dram
- top_perf_samples_per_sec: 22.2503
- top_perf_time_ms:         44.9433
- dram_time_ms:             29.9622
- compute_time_ms_lofi:     0.9598
- compute_time_ms_hifi2:    1.9197
- compute_time_ms_hifi3:    2.8795
- compute_time_ms_hifi4:    3.8393

## Files changed
- tests/benchmark/test_llms.py (added test_sip_jmed_llm_3_8x13b_op_32k_r0_1_gguf)
- tests/benchmark/benchmarks/llm_benchmark.py (fixed hasattr guard for get_weight_dtype_config_path)
- .github/workflows/perf-bench-matrix.json (added sip_jmed_llm_3_8x13b_op_32k_r0_1_gguf entry)

## tt-forge-models submodule
no change
