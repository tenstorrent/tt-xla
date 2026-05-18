loader_path: third_party.tt_forge_models.alpamayo_1_5.causal_lm.pytorch.loader
variant_id: 1_5_10B
arch: n150
status: DONE_PASS
test_function: test_alpamayo_1_5_10b
samples_per_second: 9.797
ttft_ms: 997.03
prefill_pcc: 0.996679
first_decode_pcc: 0.998746
top_perf_samples_per_sec: 23.6560
pct_of_target: 41.4
roofline_bound: dram
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: bfp_bf8
failure_reason: null

# Benchmark added: test_alpamayo_1_5_10b

## Test
tests/benchmark/test_llms.py::test_alpamayo_1_5_10b

## Model
- HF name:    nvidia/Alpamayo-1.5-10B (falls back to Qwen/Qwen3-VL-8B-Instruct)
- Loader:     third_party.tt_forge_models.alpamayo_1_5.causal_lm.pytorch.loader
- Variant:    ModelVariant.ALPAMAYO_1_5_10B

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: bfp_bf8
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  9.797
- TTFT (ms):          997.03
- Prefill PCC:        0.996679
- First decode PCC:   0.998746
- Wall clock:         0:25:08
- Hardware:           n150

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_alpamayo_1_5_10b_perf_metrics_1.json
Achieved vs top_perf_samples_per_sec: 41.4% (9.797 / 23.656)

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
- total_flops:             484358238208
- breakdown.matmul:        484358238208
- breakdown.linear:        0
- breakdown.conv2d:        0
- breakdown.sparse_matmul: 0

### Inputs
- count:        34
- memory_bytes: 134

### KV cache
- count:        301989888
- memory_bytes: 603979776
- memory_gb:    0.5625

### Params
- count:                  8190742087
- effective_count:        7568406084
- memory_bytes:           9286405784
- memory_gb:              8.648639343678951
- effective_memory_bytes: 8041721484
- effective_memory_gb:    7.489436756819487
- embedding_count:        622336003
- embedding_memory_bytes: 1244684300

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
- tests/benchmark/test_llms.py (added test_alpamayo_1_5_10b)
- tests/benchmark/benchmarks/llm_benchmark.py (general fix: VLM loader processor fallback for tokenizer)
- tests/benchmark/llm_utils/decode_utils.py (general fix: get_text_config() for VLM configs in init_static_cache)
- .github/workflows/perf-bench-matrix.json (added alpamayo_1_5_10b entry)

## tt-forge-models submodule
no change
