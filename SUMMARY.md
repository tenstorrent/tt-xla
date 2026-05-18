loader_path: third_party.tt_forge_models.deepseek_r1_0528_qwen3_8b_gguf.causal_lm.pytorch.loader
variant_id: 8B_GGUF
arch: n150
status: DONE_PASS
test_function: test_deepseek_r1_0528_qwen3_8b_gguf
samples_per_second: 13.215253897639617
ttft_ms: 809.331122
prefill_pcc: 0.998643
first_decode_pcc: 0.998109
top_perf_samples_per_sec: 23.6560
pct_of_target: 55.9
roofline_bound: dram
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: bfp_bf8
failure_reason: null

# Benchmark added: test_deepseek_r1_0528_qwen3_8b_gguf

## Test
tests/benchmark/test_llms.py::test_deepseek_r1_0528_qwen3_8b_gguf

## Model
- HF name:    lmstudio-community/DeepSeek-R1-0528-Qwen3-8B-GGUF
- Loader:     third_party.tt_forge_models.deepseek_r1_0528_qwen3_8b_gguf.causal_lm.pytorch.loader
- Variant:    ModelVariant.DEEPSEEK_R1_0528_QWEN3_8B_GGUF (= "8B_GGUF")

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: bfp_bf8
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  13.215253897639617
- TTFT (ms):          809.331122
- Prefill PCC:        0.998643
- First decode PCC:   0.998109
- Wall clock:         0:20:21
- Hardware:           n150 (wormhole_b0, n300 board used as single-chip n150)

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_deepseek_r1_0528_qwen3_8b_gguf_perf_metrics_1.json
Achieved vs top_perf_samples_per_sec: 55.9% (13.22 / 23.66)

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
- total_flops:             484358226048
- breakdown.matmul:        484358226048
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
- count:                  8190735555
- effective_count:        7568405699
- memory_bytes:           9286380296
- memory_gb:              8.64861560612917
- effective_memory_bytes: 8041720584
- effective_memory_gb:    7.4894359186291695
- embedding_count:        622329856
- embedding_memory_bytes: 1244659712

### Roofline
- bound:                    dram
- top_perf_samples_per_sec: 23.6560
- top_perf_time_ms:         42.2726
- dram_time_ms:             28.1818
- compute_time_ms_lofi:     1.8920
- compute_time_ms_hifi2:    3.7840
- compute_time_ms_hifi3:    5.6761
- compute_time_ms_hifi4:    7.5681

## Files changed
- tests/benchmark/test_llms.py (new test function test_deepseek_r1_0528_qwen3_8b_gguf)
- tests/benchmark/benchmarks/llm_benchmark.py (hasattr guard for get_weight_dtype_config_path)
- .github/workflows/perf-bench-matrix.json (new deepseek_r1_0528_qwen3_8b_gguf entry)

## tt-forge-models submodule
no change
