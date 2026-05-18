loader_path: third_party.tt_forge_models.bartowski_smallthinker_3b_preview_gguf.causal_lm.pytorch.loader
variant_id: SmallThinker_3B_Preview_GGUF
arch: p150
status: DONE_PASS
test_function: test_bartowski_smallthinker_3b_preview_gguf
samples_per_second: 40.99
ttft_ms: 222.81
prefill_pcc: 0.994550
first_decode_pcc: 0.998498
top_perf_samples_per_sec: 104.6961
pct_of_target: 39.2
roofline_bound: dram
optimization_level: 1
trace_enabled: true
experimental_weight_dtype: bfp_bf8
failure_reason: null

# Benchmark added: test_bartowski_smallthinker_3b_preview_gguf

## Test
tests/benchmark/test_llms.py::test_bartowski_smallthinker_3b_preview_gguf

## Model
- HF name:    bartowski/SmallThinker-3B-Preview-GGUF
- Loader:     third_party.tt_forge_models.bartowski_smallthinker_3b_preview_gguf.causal_lm.pytorch.loader
- Variant:    SmallThinker_3B_Preview_GGUF

## Test config landed
- optimization_level:        1
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

Note: optimization_level=2 fails with compiler error: ttnn.paged_update_cache expects sharded
input tensor (OperationValidationAndFallback, tested up to 10000 combinations). Falling back
to optimization_level=1 which passes with good PCC on the full model.

## Measured (full model, defaults)
- Sample per second:  40.99
- TTFT (ms):          222.81
- Prefill PCC:        0.994550
- First decode PCC:   0.998498
- Wall clock:         0:03:33
- Hardware:           p150

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_bartowski_smallthinker_3b_preview_gguf_perf_metrics_1.json
Achieved vs top_perf_samples_per_sec: 39.2% (40.99 / 104.70)

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
- total_flops:             197487558784
- breakdown.matmul:        185405014144
- breakdown.linear:        12082544640
- breakdown.conv2d:        0
- breakdown.sparse_matmul: 0

### Inputs
- count:        33
- memory_bytes: 132

### KV cache
- count:        75497472
- memory_bytes: 150994944
- memory_gb:    0.140625

### Params
- count:                  3397103811
- effective_count:        3085938883
- memory_bytes:           3901367048
- memory_gb:              3.633431203663349
- effective_memory_bytes: 3279037192
- effective_memory_gb:    3.053841359913349
- embedding_count:        311164928
- embedding_memory_bytes: 622329856

### Roofline
- bound:                    dram
- top_perf_samples_per_sec: 104.6961
- top_perf_time_ms:         9.5515
- dram_time_ms:             6.3676
- compute_time_ms_lofi:     0.2244
- compute_time_ms_hifi2:    0.4488
- compute_time_ms_hifi3:    0.6733
- compute_time_ms_hifi4:    0.8977

## Files changed
- tests/benchmark/test_llms.py (added test_bartowski_smallthinker_3b_preview_gguf)
- tests/benchmark/benchmarks/llm_benchmark.py (general fix: added hasattr guard for get_weight_dtype_config_path)

## tt-forge-models submodule
no change
