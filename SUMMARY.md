loader_path: third_party.tt_forge_models.deephermes_3_llama_3_3b_preview_gguf.causal_lm.pytorch.loader
variant_id: 3B_PREVIEW_GGUF
arch: p150
status: DONE_PASS
test_function: test_deephermes_3_llama_3_3b_preview_gguf
samples_per_second: 58.995
ttft_ms: 196.94
prefill_pcc: 0.998524
first_decode_pcc: 0.998707
top_perf_samples_per_sec: 96.0050
pct_of_target: 61.4
roofline_bound: dram
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: bfp_bf8
failure_reason: null

# Benchmark added: test_deephermes_3_llama_3_3b_preview_gguf

## Test
tests/benchmark/test_llms.py::test_deephermes_3_llama_3_3b_preview_gguf

## Model
- HF name:    bartowski/NousResearch_DeepHermes-3-Llama-3-3B-Preview-GGUF
- Loader:     third_party.tt_forge_models.deephermes_3_llama_3_3b_preview_gguf.causal_lm.pytorch.loader
- Variant:    ModelVariant.DEEPHERMES_3_LLAMA_3_3B_PREVIEW_GGUF (3B_PREVIEW_GGUF)

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  58.995
- TTFT (ms):          196.94
- Prefill PCC:        0.998524
- First decode PCC:   0.998707
- Wall clock:         0:06:43
- Hardware:           p150

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_deephermes_3_llama_3_3b_preview_gguf_perf_metrics_1.json
Achieved vs top_perf_samples_per_sec: 61.4% (58.995 / 96.0050)

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
- total_flops:             205604782208
- breakdown.matmul:        205604782208
- breakdown.linear:        0
- breakdown.conv2d:        0
- breakdown.sparse_matmul: 0

### Inputs
- count:        33
- memory_bytes: 132

### KV cache
- count:        234881024
- memory_bytes: 469762048
- memory_gb:    0.4375

### Params
- count:                  3606752451
- effective_count:        3212750019
- memory_bytes:           4201716488
- memory_gb:              3.9131534174084663
- effective_memory_bytes: 3413711624
- effective_memory_gb:    3.1792666986584663
- embedding_count:        394002432
- embedding_memory_bytes: 788004864

### Roofline
- bound:                    dram
- top_perf_samples_per_sec: 96.0050
- top_perf_time_ms:         10.4161
- dram_time_ms:             6.9441
- compute_time_ms_lofi:     0.2336
- compute_time_ms_hifi2:    0.4673
- compute_time_ms_hifi3:    0.7009
- compute_time_ms_hifi4:    0.9346

## Files changed
- tests/benchmark/test_llms.py (new test function added)
- .github/workflows/perf-bench-matrix.json (new entry added)

## tt-forge-models submodule
no change
