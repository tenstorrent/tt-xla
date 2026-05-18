loader_path: third_party.tt_forge_models.chandra_ocr_gguf.causal_lm.pytorch.loader
variant_id: chandra_OCR_GGUF
arch: p150
status: DONE_PASS
test_function: test_chandra_ocr_gguf
samples_per_second: 19.413077872702516
ttft_ms: 536.520123
prefill_pcc: 0.997880
first_decode_pcc: 0.998534
top_perf_samples_per_sec: 42.0551
pct_of_target: 46.2
roofline_bound: dram
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: "bfp_bf8"
failure_reason: null

# Benchmark added: test_chandra_ocr_gguf

## Test
tests/benchmark/test_llms.py::test_chandra_ocr_gguf

## Model
- HF name:    datalab-to/chandra
- Loader:     third_party.tt_forge_models.chandra_ocr_gguf.causal_lm.pytorch.loader
- Variant:    ModelVariant.CHANDRA_OCR_GGUF

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  19.413077872702516
- TTFT (ms):          536.520123
- Prefill PCC:        0.997880
- First decode PCC:   0.998534
- Wall clock:         ~0:13:31
- Hardware:           p150

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_chandra_ocr_gguf_perf_metrics_1.json
Achieved vs top_perf_samples_per_sec: 46.2% (19.41 / 42.06 samples/sec)

Note: Gap below 50% of roofline. Model is DRAM-bound. With all defaults enabled
(optimization_level=2, trace=True, bfp_bf8), further gains would require
compiler-side improvements.

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

## Infrastructure fixes (general, not model-specific)

Two general fixes were made to `tests/benchmark/benchmarks/llm_benchmark.py`:

1. **`get_text_config()` helper**: Vision-language models (e.g. Qwen3VLForConditionalGeneration)
   have a top-level VL config (Qwen3VLConfig) whose text-decoder attributes live in
   a nested `text_config`. Passing the VL config to cache-init helpers that expect
   `hidden_size` / `num_attention_heads` fails with AttributeError. Added `get_text_config()`
   to extract `config.text_config` when available, used in all `construct_inputs` calls
   and the `num_hidden_layers` extraction.

2. **`get_weight_dtype_config_path()` guard**: The harness called this method unconditionally,
   but no current loader implements it. Changed `else:` to `elif hasattr(model_loader, ...):`
   so loaders without this method are silently skipped.

## Files changed
- tests/benchmark/test_llms.py
- tests/benchmark/benchmarks/llm_benchmark.py
- .github/workflows/perf-bench-matrix.json

## tt-forge-models submodule
no change
