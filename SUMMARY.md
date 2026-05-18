loader_path: third_party.tt_forge_models.gemma3_270m_gguf.causal_lm.pytorch.loader
variant_id: 270M_IT_GGUF
arch: p150
status: DONE_PASS
test_function: test_gemma3_270m_gguf
samples_per_second: 100.11788453316981
ttft_ms: 167.498179
prefill_pcc: 0.965058
first_decode_pcc: 0.974359
top_perf_samples_per_sec: 1082.2068
pct_of_target: 9.3
roofline_bound: dram
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: "bfp_bf8"
failure_reason: null

# Benchmark added: gemma3_270m_gguf

## Test
tests/benchmark/test_llms.py::test_gemma3_270m_gguf

## Model
- HF name:    unsloth/gemma-3-270m-it-GGUF
- Loader:     third_party.tt_forge_models.gemma3_270m_gguf.causal_lm.pytorch.loader
- Variant:    ModelVariant.GEMMA_3_270M_IT_GGUF

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  100.12
- TTFT (ms):          167.498
- Prefill PCC:        0.965058
- First decode PCC:   0.974359
- Wall clock:         0:05:46
- Hardware:           p150

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_gemma3_270m_gguf_perf_metrics_2.json
Achieved vs top_perf_samples_per_sec: 9.3% (100.12 / 1082.21)

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
- total_flops:             17154703616
- breakdown.matmul:        17154703616
- breakdown.linear:        0
- breakdown.conv2d:        0
- breakdown.sparse_matmul: 0

### Inputs
- count:        33
- memory_bytes: 132

### KV cache
- count:        37748736
- memory_bytes: 75497472
- memory_gb:    0.0703125

### Params
- count:                  435870597
- effective_count:        268098437
- memory_bytes:           620563982
- memory_gb:              0.5779452454298735
- effective_memory_bytes: 285019662
- effective_memory_gb:    0.26544524542987347
- embedding_count:        167772160
- embedding_memory_bytes: 335544320

### Roofline
- bound:                    dram
- top_perf_samples_per_sec: 1082.2068
- top_perf_time_ms:         0.9240
- dram_time_ms:             0.6160
- compute_time_ms_lofi:     0.0195
- compute_time_ms_hifi2:    0.0390
- compute_time_ms_hifi3:    0.0585
- compute_time_ms_hifi4:    0.0780

## Files changed
- tests/benchmark/test_llms.py (added test_gemma3_270m_gguf)
- .github/workflows/perf-bench-matrix.json (added unsloth_gemma-3-270m-it-gguf entry)
- tests/benchmark/benchmarks/llm_benchmark.py (two general infra fixes: sync per-layer attention_type when overriding config.layer_types for Gemma3 sliding attention compatibility; add hasattr guard for get_weight_dtype_config_path)

## tt-forge-models submodule
no change

## Infrastructure fixes
1. **Gemma3 sliding attention fix** (general): `setup_model_and_tokenizer` now syncs `attention_type` and `layer_type` per-module when overriding `config.layer_types` to `full_attention`. Without this, Gemma3 forward raises `KeyError: 'sliding_attention'` because decoder layers retain their original `attention_type` while the config is overridden, making `position_embeddings[decoder_layer.attention_type]` fail.
2. **`get_weight_dtype_config_path` guard** (general): Changed `else:` to `elif hasattr(model_loader, "get_weight_dtype_config_path"):` so loaders without this method (like Gemma3's) don't raise `AttributeError`.
