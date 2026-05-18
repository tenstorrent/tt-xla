loader_path: third_party.tt_forge_models.h2oai_llama2_0b_unit_test.causal_lm.pytorch.loader
variant_id: h2oai_llama2_0b_unit_test
arch: p150
status: DONE_PASS
test_function: test_h2oai_llama2_0b_unit_test
samples_per_second: 984.78
ttft_ms: 11.27
prefill_pcc: 0.999944
first_decode_pcc: 0.999961
top_perf_samples_per_sec: 567072.3671
pct_of_target: 0.2
roofline_bound: dram
optimization_level: 1
trace_enabled: true
experimental_weight_dtype: "bfp_bf8"
failure_reason: null

# Benchmark added: h2oai_llama2_0b_unit_test

## Test
tests/benchmark/test_llms.py::test_h2oai_llama2_0b_unit_test

## Model
- HF name:    h2oai/llama2-0b-unit-test
- Loader:     third_party.tt_forge_models.h2oai_llama2_0b_unit_test.causal_lm.pytorch.loader
- Variant:    ModelVariant.H2OAI_LLAMA2_0B_UNIT_TEST

## Test config landed
- optimization_level:        1
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

Note: optimization_level=2 triggers TT_THROW: No circular buffer with id 0 exists in Program.

## Measured (full model, defaults)
- Sample per second:  984.78
- TTFT (ms):          11.27
- Prefill PCC:        0.999944
- First decode PCC:   0.999961
- Wall clock:         0:00:18
- Hardware:           p150

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_h2oai_llama2_0b_unit_test_perf_metrics_0.json
Achieved vs top_perf_samples_per_sec: 0.2% (984.78 / 567072.37)

Note: gap is expected — this is a ~771k-param unit-test model; kernel dispatch
overhead completely dominates, making the DRAM roofline a theoretical floor.

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
- total_flops:             445685868
- breakdown.matmul:        445685868
- breakdown.linear:        0
- breakdown.conv2d:        0
- breakdown.sparse_matmul: 0

### Inputs
- count:        594
- memory_bytes: 2376

### KV cache
- count:        196608
- memory_bytes: 393216
- memory_gb:    0.0003662109375

### Params
- count:                  771074
- effective_count:        387074
- memory_bytes:           1179712
- memory_gb:              0.0010986924171447754
- effective_memory_bytes: 411712
- effective_memory_gb:    0.0003834366798400879
- embedding_count:        384000
- embedding_memory_bytes: 768000

### Roofline
- bound:                    dram
- top_perf_samples_per_sec: 567072.3671
- top_perf_time_ms:         0.0018
- dram_time_ms:             0.0012
- compute_time_ms_lofi:     0.0005
- compute_time_ms_hifi2:    0.0010
- compute_time_ms_hifi3:    0.0015
- compute_time_ms_hifi4:    0.0020

## Files changed
- tests/benchmark/test_llms.py (added test_h2oai_llama2_0b_unit_test)
- .github/workflows/perf-bench-matrix.json (added h2oai_llama2_0b_unit_test entry)
- tests/benchmark/benchmarks/llm_benchmark.py (general fix: guard get_weight_dtype_config_path with hasattr)

## tt-forge-models submodule
no change
