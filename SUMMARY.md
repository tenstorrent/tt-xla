loader_path: third_party.tt_forge_models.alpaca.causal_lm.pytorch.loader
variant_id: native
arch: n150
status: DONE_PASS
test_function: test_alpaca_native
samples_per_second: 2.073
ttft_ms: 2966.7
prefill_pcc: 0.986504
first_decode_pcc: 0.993611
top_perf_samples_per_sec: 24.239
pct_of_target: 8.6
roofline_bound: dram
optimization_level: 0
trace_enabled: true
experimental_weight_dtype: bfp_bf8
failure_reason: null

# Benchmark added: test_alpaca_native

## Test
tests/benchmark/test_llms.py::test_alpaca_native

## Model
- HF name:    maicomputer/alpaca-native
- Loader:     third_party.tt_forge_models.alpaca.causal_lm.pytorch.loader
- Variant:    ModelVariant.ALPACA_NATIVE ("native")

## Test config landed
- optimization_level:        0
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

Note: optimization_level=0 is required; levels 1 and 2 both hang on n150 for this 7B LLaMA-based model.

## Measured (full model, defaults)
- Sample per second:  2.073
- TTFT (ms):          2966.7
- Prefill PCC:        0.986504
- First decode PCC:   0.993611
- Wall clock:         ~0:06:00
- Hardware:           n150 (wormhole_b0, single chip of n300)

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_alpaca_native_perf_metrics_1.json
Achieved vs top_perf_samples_per_sec: 8.6% (2.073 / 24.239)

Note: Low % of roofline is expected at optimization_level=0 (all tensors in DRAM, no SRAM push).

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
- total_flops:             425000697984
- breakdown.matmul:        425000697984
- breakdown.linear:        0
- breakdown.conv2d:        0
- breakdown.sparse_matmul: 0

### Inputs
- count:        33
- memory_bytes: 132

### KV cache
- count:        1073741824
- memory_bytes: 2147483648
- memory_gb:    2

### Params
- count:                  6738424006
- effective_count:        6607347910
- memory_bytes:           7282709524
- memory_gb:              6.782551784068346
- effective_memory_bytes: 7020557332
- effective_memory_gb:    6.538403529673815
- embedding_count:        131076096
- embedding_memory_bytes: 262152192

### Roofline
- bound:                    dram
- top_perf_samples_per_sec: 24.2390
- top_perf_time_ms:         41.2559
- dram_time_ms:             27.5039
- compute_time_ms_lofi:     1.6602
- compute_time_ms_hifi2:    3.3203
- compute_time_ms_hifi3:    4.9805
- compute_time_ms_hifi4:    6.6406

## Files changed
- tests/benchmark/test_llms.py (added test_alpaca_native)
- tests/benchmark/benchmarks/llm_benchmark.py (fix: guard get_weight_dtype_config_path with hasattr for loaders that don't implement it)
- .github/workflows/perf-bench-matrix.json (added alpaca_native entry)

## tt-forge-models submodule
no change
