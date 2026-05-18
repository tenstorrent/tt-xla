loader_path: third_party.tt_forge_models.alpamayo_1_5.causal_lm.pytorch.loader
variant_id: 1_5_10B
arch: p150
status: DONE_PASS
test_function: test_alpamayo_1_5_10b
samples_per_second: 19.75
ttft_ms: 494.45
prefill_pcc: 0.987258
first_decode_pcc: 0.998458
top_perf_samples_per_sec: 42.0551
pct_of_target: 47.0
roofline_bound: dram
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: "bfp_bf8"
failure_reason: null

# Benchmark added: test_alpamayo_1_5_10b

## Test
tests/benchmark/test_llms.py::test_alpamayo_1_5_10b

## Model
- HF name:    Qwen/Qwen3-VL-8B-Instruct (fallback from nvidia/Alpamayo-1.5-10B)
- Loader:     third_party.tt_forge_models.alpamayo_1_5.causal_lm.pytorch.loader
- Variant:    ModelVariant.ALPAMAYO_1_5_10B (= "1_5_10B")

Note: The loader falls back to Qwen/Qwen3-VL-8B-Instruct (the public VLM backbone)
because the full Alpamayo-1.5-10B model requires a proprietary `alpamayo1_5` package
and a gated NVIDIA model. Two general infrastructure fixes were needed to support
VLM-based loaders in the benchmark harness:
  1. `setup_model_and_tokenizer` in `llm_benchmark.py`: fall back to
     `model_loader.processor.tokenizer` when `model_loader.tokenizer` is not set.
  2. `init_static_cache` in `llm_utils/decode_utils.py`: use `config.text_config` when
     the top-level config is a VLM config (e.g. Qwen3VLConfig) that stores LM
     attributes under `text_config`.
  3. `benchmark_llm_torch_xla` in `llm_benchmark.py`: guard
     `get_weight_dtype_config_path()` call with `hasattr` check, matching the pattern
     used in `dynamic_torch_model_tester.py`.

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  19.75
- TTFT (ms):          494.45
- Prefill PCC:        0.987258
- First decode PCC:   0.998458
- Wall clock:         0:12:26
- Hardware:           p150 (blackhole)

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_alpamayo_1_5_10b_perf_metrics_1.json
Achieved vs top_perf_samples_per_sec: 47.0% (19.75 / 42.0551)

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
- top_perf_samples_per_sec: 42.0551
- top_perf_time_ms:         23.7784
- dram_time_ms:             15.8522
- compute_time_ms_lofi:     0.5504
- compute_time_ms_hifi2:    1.1008
- compute_time_ms_hifi3:    1.6512
- compute_time_ms_hifi4:    2.2016

## Files changed
- tests/benchmark/test_llms.py (added test_alpamayo_1_5_10b)
- tests/benchmark/benchmarks/llm_benchmark.py (general VLM tokenizer fallback; hasattr guard for get_weight_dtype_config_path)
- tests/benchmark/llm_utils/decode_utils.py (general VLM text_config fallback in init_static_cache)

## tt-forge-models submodule
no change
