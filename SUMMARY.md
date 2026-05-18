loader_path: third_party.tt_forge_models.cpm_generate.causal_lm.pytorch.loader
variant_id: Default
arch: p150
status: DONE_PASS
test_function: test_cpm_generate
samples_per_second: 3.330629280200393
ttft_ms: 1644.551956
prefill_pcc: 0.999331
first_decode_pcc: 0.999688
top_perf_samples_per_sec: 101.3584
pct_of_target: 3.3
roofline_bound: dram
optimization_level: 0
trace_enabled: true
experimental_weight_dtype: null
failure_reason: null

# Benchmark added: test_cpm_generate

## Test
tests/benchmark/test_llms.py::test_cpm_generate

## Model
- HF name:    TsinghuaAI/CPM-Generate
- Loader:     third_party.tt_forge_models.cpm_generate.causal_lm.pytorch.loader
- Variant:    ModelVariant.BASE ("Default")

## Test config landed
- optimization_level:        0
- trace_enabled:             true
- experimental_weight_dtype: none
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

Note: optimization_level=1 and 2 both fail first decode PCC (~0.86 and ~0.78
respectively) due to GPT2 Conv1D numerical precision issues at higher
optimization levels. Level 0 passes with PCC=0.999688.
experimental_weight_dtype disabled (bfp_bf8 caused PCC regression to ~0.78).

## Measured (full model, defaults)
- Sample per second:  3.330629280200393
- TTFT (ms):          1644.551956
- Prefill PCC:        0.999331
- First decode PCC:   0.999688
- Wall clock:         0:03:19 (199.03s)
- Hardware:           p150

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_cpm_generate_perf_metrics_3.json
Achieved vs top_perf_samples_per_sec: 3.3% (3.33 / 101.36 samples/sec)

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
- total_flops:             167342243840
- breakdown.matmul:        6257377280
- breakdown.linear:        161084866560
- breakdown.conv2d:        0
- breakdown.sparse_matmul: 0

### Inputs
- count:        33
- memory_bytes: 132

### KV cache
- count:        671088640
- memory_bytes: 1342177280
- memory_gb:    1.25

### Params
- count:                  2673874055
- effective_count:        2594452615
- memory_bytes:           10382387736
- memory_gb:              9.669352076947689
- effective_memory_bytes: 10223544856
- effective_memory_gb:    9.521418116986752
- embedding_count:        79421440
- embedding_memory_bytes: 158842880

### Roofline
- bound:                    dram
- top_perf_samples_per_sec: 101.3584
- top_perf_time_ms:         9.8660
- dram_time_ms:             6.5773
- compute_time_ms_lofi:     0.1902
- compute_time_ms_hifi2:    0.3803
- compute_time_ms_hifi3:    0.5705
- compute_time_ms_hifi4:    0.7606

## Files changed
- tests/benchmark/test_llms.py (added test_cpm_generate)
- tests/benchmark/benchmarks/llm_benchmark.py (two general harness fixes:
  1. lazy tokenizer loading via _load_tokenizer() when tokenizer is None after load_model();
  2. graceful skip of get_weight_dtype_config_path() when method is absent)

## tt-forge-models submodule
no change
