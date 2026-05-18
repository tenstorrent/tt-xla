loader_path: third_party.tt_forge_models.distilgpt2.causal_lm.pytorch.loader
variant_id: DistilGPT2
arch: p150
status: DONE_PASS
test_function: test_distilgpt2
samples_per_second: 285.14
ttft_ms: 43.30
prefill_pcc: 0.998222
first_decode_pcc: 0.992019
top_perf_samples_per_sec: 2784.3522
pct_of_target: 10.2
roofline_bound: dram
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: bfp_bf8
failure_reason: null

# Benchmark added: distilgpt2

## Test
tests/benchmark/test_llms.py::test_distilgpt2

## Model
- HF name:    distilgpt2
- Loader:     third_party.tt_forge_models.distilgpt2.causal_lm.pytorch.loader
- Variant:    ModelVariant.DISTILGPT2

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  285.14
- TTFT (ms):          43.30
- Prefill PCC:        0.998222
- First decode PCC:   0.992019
- Wall clock:         0:01:09
- Hardware:           p150

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_distilgpt2_perf_metrics_0.json
Achieved vs top_perf_samples_per_sec: 10.2%

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
- total_flops:             88220958720
- breakdown.matmul:        41993945088
- breakdown.linear:        46227013632
- breakdown.conv2d:        0
- breakdown.sparse_matmul: 0

### Inputs
- count:        561
- memory_bytes: 2244

### KV cache
- count:        37748736
- memory_bytes: 75497472
- memory_gb:    0.0703125

### Params
- count:                  120510084
- effective_count:        81126276
- memory_bytes:           165105212
- memory_gb:              0.1537662111222744
- effective_memory_bytes: 86337596
- effective_memory_gb:    0.08040815219283104
- embedding_count:        39383808
- embedding_memory_bytes: 78767616

### Roofline
- bound:                    dram
- top_perf_samples_per_sec: 2784.3522
- top_perf_time_ms:         0.3591
- dram_time_ms:             0.2394
- compute_time_ms_lofi:     0.1003
- compute_time_ms_hifi2:    0.2005
- compute_time_ms_hifi3:    0.3008
- compute_time_ms_hifi4:    0.4010

## Files changed
- tests/benchmark/test_llms.py (added test_distilgpt2)
- tests/benchmark/benchmarks/llm_benchmark.py (fixed get_weight_dtype_config_path hasattr guard)

## tt-forge-models submodule
no change
