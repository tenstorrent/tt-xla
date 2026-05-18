loader_path: third_party.tt_forge_models.gpt2_genre_story_generator.causal_lm.pytorch.loader
variant_id: default
arch: p150
status: DONE_PASS
test_function: test_gpt2_genre_story_generator
samples_per_second: 165.76
ttft_ms: 80.73
prefill_pcc: 0.998968
first_decode_pcc: 0.997762
top_perf_samples_per_sec: 1661.9472
pct_of_target: 10.0
roofline_bound: dram
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: bfp_bf8
failure_reason: null

# Benchmark added: gpt2_genre_story_generator

## Test
tests/benchmark/test_llms.py::test_gpt2_genre_story_generator

## Model
- HF name:    pranavpsv/gpt2-genre-story-generator
- Loader:     third_party.tt_forge_models.gpt2_genre_story_generator.causal_lm.pytorch.loader
- Variant:    ModelVariant.DEFAULT

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  165.76
- TTFT (ms):          80.73
- Prefill PCC:        0.998968
- First decode PCC:   0.997762
- Wall clock:         0:02:04
- Hardware:           p150

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_gpt2_genre_story_generator_perf_metrics_0.json
Achieved vs top_perf_samples_per_sec: 10.0% (165.76 / 1661.95)

Note: GPT-2 is a very small model (~163M params, ~0.14 GB effective weight memory).
At this scale, host-side overhead (kernel launch, synchronization, data transfer)
dominates the compute/DRAM time, explaining the large gap from the theoretical roofline.

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
- total_flops:             134455492608
- breakdown.matmul:        42001465344
- breakdown.linear:        92454027264
- breakdown.conv2d:        0
- breakdown.sparse_matmul: 0

### Inputs
- count:        561
- memory_bytes: 2244

### KV cache
- count:        75497472
- memory_bytes: 150994944
- memory_gb:    0.140625

### Params
- count:                  163051140
- effective_count:        123660420
- memory_bytes:           210450668
- memory_gb:              0.1959974579513073
- effective_memory_bytes: 131669228
- effective_memory_gb:    0.12262652441859245
- embedding_count:        39390720
- embedding_memory_bytes: 78781440

### Roofline
- bound:                    dram
- top_perf_samples_per_sec: 1661.9472
- top_perf_time_ms:         0.6017
- dram_time_ms:             0.4011
- compute_time_ms_lofi:     0.1528
- compute_time_ms_hifi2:    0.3056
- compute_time_ms_hifi3:    0.4584
- compute_time_ms_hifi4:    0.6112

## Files changed
- tests/benchmark/test_llms.py (added test_gpt2_genre_story_generator)
- tests/benchmark/benchmarks/llm_benchmark.py (fixed get_weight_dtype_config_path to use hasattr guard)
- .github/workflows/perf-bench-matrix.json (added gpt2_genre_story_generator entry)

## tt-forge-models submodule
no change
