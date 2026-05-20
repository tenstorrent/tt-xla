loader_path: third_party.tt_forge_models.japanese_stablelm_instruct_gamma_gguf.causal_lm.pytorch.loader
variant_id: 7B
arch: p150
status: DONE_PASS
test_function: test_japanese_stablelm_instruct_gamma_gguf_7b
samples_per_second: 4.614798636265359
ttft_ms: 789.219983
prefill_pcc: 0.976313
first_decode_pcc: 0.998858
top_perf_samples_per_sec: 44.855075772401314
pct_of_target: 10.3
roofline_bound: dram
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: bfp_bf8
failure_reason: null

# Benchmark added: test_japanese_stablelm_instruct_gamma_gguf_7b

## Test
tests/benchmark/test_llms.py::test_japanese_stablelm_instruct_gamma_gguf_7b

## Model
- HF name:    TheBloke/japanese-stablelm-instruct-gamma-7B-GGUF
- Loader:     third_party.tt_forge_models.japanese_stablelm_instruct_gamma_gguf.causal_lm.pytorch.loader
- Variant:    ModelVariant.JAPANESE_STABLELM_INSTRUCT_GAMMA_7B (= "7B")

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  4.614798636265359
- TTFT (ms):          789.219983
- Prefill PCC:        0.976313
- First decode PCC:   0.998858
- Wall clock:         0:44:35
- Hardware:           p150

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_japanese_stablelm_instruct_gamma_gguf_7b_perf_metrics_1.json
Achieved vs top_perf_samples_per_sec: 10.3% (4.61 / 44.86 samples/sec)

### System
- arch:                        blackhole
- chip_count_in_system_desc:   1
- single_chip_assumption:      True
- worker_grid_cores:           130
- dram_bandwidth_bytes_per_sec: 512000000000

### Peak FLOPs
- lofi:  1040000000000000
- hifi2: 520000000000000
- hifi3: 346666666666666
- hifi4: 260000000000000

### Compute
- total_flops:             457212690560
- breakdown.matmul:        457212690560
- breakdown.linear:        0
- breakdown.conv2d:        0
- breakdown.sparse_matmul: 0

### Inputs
- count:        33
- memory_bytes: 132

### KV cache
- count:        268435456
- memory_bytes: 536870912
- memory_gb:    0.5

### Params
- count:                  7241732294
- effective_count:        7110660294
- memory_bytes:           7817470740
- memory_gb:              7.28058697655797
- effective_memory_bytes: 7555326740
- effective_memory_gb:    7.03644635155797
- embedding_count:        131072000
- embedding_memory_bytes: 262144000

### Roofline
- bound:                    dram
- top_perf_samples_per_sec: 44.855075772401314
- top_perf_time_ms:         22.2940
- dram_time_ms:             14.8627
- compute_time_ms_lofi:     0.4396
- compute_time_ms_hifi2:    0.8793
- compute_time_ms_hifi3:    1.3189
- compute_time_ms_hifi4:    1.7585

## Files changed
- tests/benchmark/test_llms.py
- tests/benchmark/benchmarks/llm_benchmark.py
- .github/workflows/perf-bench-matrix.json

## tt-forge-models submodule
no change
