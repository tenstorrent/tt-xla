loader_path: third_party.tt_forge_models.pythia.causal_lm.pytorch.loader
variant_id: 31M
arch: p150
status: DONE_FAIL
test_function: test_pythia_31m
samples_per_second: null
ttft_ms: null
prefill_pcc: null
first_decode_pcc: null
top_perf_samples_per_sec: 10959.9987
pct_of_target: null
roofline_bound: dram
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: null
failure_reason: "Full model prefill PCC=0.568 (required 0.94); bfp_bf8 disabled since it collapses prefill PCC further; numerical accuracy degradation with GPT-NeoX parallel attention+MLP architecture across all opt_level (1,2) and trace (True/False) configurations; 1-layer decode PCC also fails (0.939 vs 0.94); root cause is compiler/runtime accuracy issue with GPTNeoX architecture"

# Benchmark added: pythia_31m

## Test
tests/benchmark/test_llms.py::test_pythia_31m

## Model
- HF name:    EleutherAI/pythia-31m
- Loader:     third_party.tt_forge_models.pythia.causal_lm.pytorch.loader
- Variant:    ModelVariant.PYTHIA_31M

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: none (bfp_bf8 disabled due to severe PCC regression)
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  null (DONE_FAIL)
- TTFT (ms):          null (DONE_FAIL)
- Prefill PCC:        0.568 (required 0.94 — FAIL)
- First decode PCC:   null (prefill failed first)
- Wall clock:         N/A
- Hardware:           p150

## PCC Diagnostic Summary
Tested configs:
| Config                              | Prefill PCC | Decode PCC |
|-------------------------------------|-------------|------------|
| 1-layer, bfp_bf8, opt=2, trace=T    | 0.958 ✓     | 0.935 ✗    |
| 1-layer, no-bfp, opt=2, trace=T     | 0.958 ✓     | 0.939 ✗    |
| 1-layer, no-bfp, opt=1, trace=T     | 0.958 ✓     | 0.936 ✗    |
| 1-layer, no-bfp, opt=2, trace=F     | 0.958 ✓     | 0.939 ✗    |
| full (6-layer), bfp_bf8, opt=2, T   | 0.551 ✗     | —          |
| full (6-layer), no-bfp, opt=2, T    | 0.568 ✗     | —          |
| full (6-layer), no-bfp, opt=1, T    | 0.533 ✗     | —          |
| full (6-layer), no-bfp, opt=2, F    | 0.568 ✗     | —          |

The 1-layer decode PCC is consistently ~0.939 (just below the 0.94 threshold).
The full 6-layer prefill PCC is consistently ~0.55, indicating a fundamental numerical
accuracy issue with GPT-NeoX's parallel attention+MLP residual architecture in the
TT-XLA compiler backend. The root cause is not optimization_level or trace settings.

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_pythia_31m_perf_metrics_0.json
Achieved vs top_perf_samples_per_sec: N/A (test failed PCC)

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
- total_flops:             19152421000
- breakdown.matmul:        14011072648
- breakdown.linear:        5141348352
- breakdown.conv2d:        0
- breakdown.sparse_matmul: 0

### Inputs
- count:        561
- memory_bytes: 2244

### KV cache
- count:        12582912
- memory_bytes: 25165824
- memory_gb:    0.0234375

### Params
- count:                  30494724
- effective_count:        17616900
- memory_bytes:           60989448
- memory_gb:              0.05680084973573685
- effective_memory_bytes: 35233800
- effective_memory_gb:    0.03281403332948685
- embedding_count:        12877824
- embedding_memory_bytes: 25755648

### Roofline
- bound:                    dram
- top_perf_samples_per_sec: 10959.9987
- top_perf_time_ms:         0.0912
- dram_time_ms:             0.0608
- compute_time_ms_lofi:     0.0218
- compute_time_ms_hifi2:    0.0435
- compute_time_ms_hifi3:    0.0653
- compute_time_ms_hifi4:    0.0871

## Files changed
- tests/benchmark/test_llms.py (added test_pythia_31m)
- tests/benchmark/benchmarks/llm_benchmark.py (general fix: hasattr guard for get_weight_dtype_config_path)

## tt-forge-models submodule
no change
