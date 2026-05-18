loader_path: third_party.tt_forge_models.c2s_scale_gemma_2_2b.causal_lm.pytorch.loader
variant_id: C2S_Scale_Gemma_2_2B
arch: p150
status: DONE_PASS
test_function: test_c2s_scale_gemma_2_2b
samples_per_second: 53.75
ttft_ms: 261.759069
prefill_pcc: 0.997264
first_decode_pcc: 0.996561
top_perf_samples_per_sec: 116.8566
pct_of_target: 46.0
roofline_bound: dram
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: "bfp_bf8"
failure_reason: null

# Benchmark added: test_c2s_scale_gemma_2_2b

## Test
tests/benchmark/test_llms.py::test_c2s_scale_gemma_2_2b

## Model
- HF name:    vandijklab/C2S-Scale-Gemma-2-2B
- Loader:     third_party.tt_forge_models.c2s_scale_gemma_2_2b.causal_lm.pytorch.loader
- Variant:    C2S_Scale_Gemma_2_2B

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  53.75
- TTFT (ms):          261.759069
- Prefill PCC:        0.997264
- First decode PCC:   0.996561
- Wall clock:         0:08:15
- Hardware:           p150

## Note on perf gap
Achieved 46.0% of roofline top_perf_samples_per_sec. All perf features are at
maximum (optimization_level=2, trace_enabled=True, experimental_weight_dtype=bfp_bf8).
The gap may reflect DRAM-bound characteristics of this 2B model at batch_size=32 on p150.

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_c2s_scale_gemma_2_2b_perf_metrics_2.json
Achieved vs top_perf_samples_per_sec: 46.0%

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
- total_flops:             167302398208
- breakdown.matmul:        167302398208
- breakdown.linear:        0
- breakdown.conv2d:        0
- breakdown.sparse_matmul: 0

### Inputs
- count:        34
- memory_bytes: 134

### KV cache
- count:        218103808
- memory_bytes: 436207616
- memory_gb:    0.40625

### Params
- count:                  3204166151
- effective_count:        2614342151
- memory_bytes:           3958097940
- memory_gb:              3.686265964061022
- effective_memory_bytes: 2778449940
- effective_memory_gb:    2.587633151561022
- embedding_count:        589824000
- embedding_memory_bytes: 1179648000

### Roofline
- bound:                    dram
- top_perf_samples_per_sec: 116.8566
- top_perf_time_ms:         8.5575
- dram_time_ms:             5.7050
- compute_time_ms_lofi:     0.1901
- compute_time_ms_hifi2:    0.3802
- compute_time_ms_hifi3:    0.5703
- compute_time_ms_hifi4:    0.7605

## Files changed
- tests/benchmark/test_llms.py (added test_c2s_scale_gemma_2_2b)
- tests/benchmark/benchmarks/llm_benchmark.py (fixed missing hasattr guard for get_weight_dtype_config_path)
- .github/workflows/perf-bench-matrix.json (added vandijklab_c2s_scale_gemma_2_2b_accuracy entry)

## tt-forge-models submodule
no change
