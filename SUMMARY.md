loader_path: third_party.tt_forge_models.lmstudio_gemma_3_4b_qat_gguf.causal_lm.pytorch.loader
variant_id: 4B_IT_QAT_GGUF
arch: p150
status: DONE_PASS
test_function: test_lmstudio_gemma_3_4b_qat_gguf
samples_per_second: 29.333282761939035
ttft_ms: 435.850898
prefill_pcc: 0.993743
first_decode_pcc: 0.988003
top_perf_samples_per_sec: 79.4603
pct_of_target: 36.9
roofline_bound: dram
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: bfp_bf8
failure_reason: null

# Benchmark added: test_lmstudio_gemma_3_4b_qat_gguf

## Test
tests/benchmark/test_llms.py::test_lmstudio_gemma_3_4b_qat_gguf

## Model
- HF name:    lmstudio-community/gemma-3-4B-it-qat-GGUF
- Loader:     third_party.tt_forge_models.lmstudio_gemma_3_4b_qat_gguf.causal_lm.pytorch.loader
- Variant:    GEMMA_3_4B_IT_QAT_GGUF (4B_IT_QAT_GGUF)

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  29.33
- TTFT (ms):          435.85
- Prefill PCC:        0.993743
- First decode PCC:   0.988003
- Wall clock:         0:17:14
- Hardware:           p150

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_lmstudio_gemma_3_4b_qat_gguf_perf_metrics_2.json
Achieved vs top_perf_samples_per_sec: 36.9% (29.33 / 79.46)

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
- total_flops:             248313282816
- breakdown.matmul:        248313282816
- breakdown.linear:        0
- breakdown.conv2d:        0
- breakdown.sparse_matmul: 0

### Inputs
- count:        33
- memory_bytes: 132

### KV cache
- count:        285212672
- memory_bytes: 570425344
- memory_gb:    0.53125

### Params
- count:                  4551515909
- effective_count:        3880263429
- memory_bytes:           5466366990
- memory_gb:              5.090950978919864
- effective_memory_bytes: 4123862030
- effective_memory_gb:    3.8406458031386137
- embedding_count:        671252480
- embedding_memory_bytes: 1342504960

### Roofline
- bound:                    dram
- top_perf_samples_per_sec: 79.4603
- top_perf_time_ms:         12.5849
- dram_time_ms:             8.3899
- compute_time_ms_lofi:     0.2822
- compute_time_ms_hifi2:    0.5643
- compute_time_ms_hifi3:    0.8465
- compute_time_ms_hifi4:    1.1287

## Files changed
- tests/benchmark/test_llms.py (added test_lmstudio_gemma_3_4b_qat_gguf)
- tests/benchmark/benchmarks/llm_benchmark.py (two general infrastructure fixes:
  1. After overriding config.layer_types, also update each decoder layer's
     attention_type attribute to prevent KeyError in mixed-attention models (e.g. Gemma3);
  2. Guard get_weight_dtype_config_path() call with hasattr() check for loaders
     that don't implement this optional method)

## tt-forge-models submodule
no change
