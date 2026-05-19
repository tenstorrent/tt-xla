loader_path: third_party.tt_forge_models.libretranslate_gemma3.causal_lm.pytorch.loader
variant_id: 1B_IT_Q4_0
arch: p150
status: DONE_PASS
test_function: test_libretranslate_gemma3_1b_it_q4_0
samples_per_second: 64.73310495983897
ttft_ms: 246.853042
prefill_pcc: 0.995641
first_decode_pcc: 0.987096
top_perf_samples_per_sec: 313.9094
pct_of_target: 20.6
roofline_bound: dram
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: bfp_bf8
failure_reason: null

# Benchmark added: test_libretranslate_gemma3_1b_it_q4_0

## Test
tests/benchmark/test_llms.py::test_libretranslate_gemma3_1b_it_q4_0

## Model
- HF name:    libretranslate/gemma3
- Loader:     third_party.tt_forge_models.libretranslate_gemma3.causal_lm.pytorch.loader
- Variant:    ModelVariant.GEMMA_3_1B_IT_Q4_0

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: bfp_bf8
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  64.73310495983897
- TTFT (ms):          246.853042
- Prefill PCC:        0.995641
- First decode PCC:   0.987096
- Wall clock:         0:08:44
- Hardware:           p150

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_libretranslate_gemma3_1b_it_q4_0_perf_metrics_2.json
Achieved vs top_perf_samples_per_sec: 20.6%

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
- total_flops:             63984108032
- breakdown.matmul:        63984108032
- breakdown.linear:        0
- breakdown.conv2d:        0
- breakdown.sparse_matmul: 0

### Inputs
- count:        34
- memory_bytes: 134

### KV cache
- count:        54525952
- memory_bytes: 109051904
- memory_gb:    0.1015625

### Params
- count:                  1301876230
- effective_count:        999886342
- memory_bytes:           1666754578
- memory_gb:              1.5522861648350954
- effective_memory_bytes: 1062774802
- effective_memory_gb:    0.9897861648350954
- embedding_count:        301989888
- embedding_memory_bytes: 603979776

### Roofline
- bound:                    dram
- top_perf_samples_per_sec: 313.9094
- top_perf_time_ms:         3.1856
- dram_time_ms:             2.1238
- compute_time_ms_lofi:     0.0727
- compute_time_ms_hifi2:    0.1454
- compute_time_ms_hifi3:    0.2181
- compute_time_ms_hifi4:    0.2908

## Files changed
- tests/benchmark/test_llms.py (new test function test_libretranslate_gemma3_1b_it_q4_0)
- tests/benchmark/benchmarks/llm_benchmark.py (general fix: remove model.config.layer_types override from setup_model_and_tokenizer that broke Gemma3 sliding_attention routing; hasattr guard for get_weight_dtype_config_path)
- tests/benchmark/llm_utils/decode_utils.py (general fix: init_static_cache now uses a copy of config with all full_attention for StaticCache creation, preserving original model config.layer_types)

## tt-forge-models submodule
no change
