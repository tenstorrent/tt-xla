loader_path: third_party.tt_forge_models.wizardlm_1_0_uncensored_llama2_13b_gguf.causal_lm.pytorch.loader
variant_id: 13B_GGUF
arch: p150
status: DONE_FAIL
test_function: test_wizardlm_1_0_uncensored_llama2_13b_gguf
samples_per_second: 10.976570052221641
ttft_ms: 635.867557
prefill_pcc: 0.999490
first_decode_pcc: 0.304799
top_perf_samples_per_sec: 22.7802
pct_of_target: 48.2
roofline_bound: dram
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: null
failure_reason: "First decode PCC catastrophically low on full model (0.305 vs 0.94 required) with both bfp_bf8 (0.308) and without (0.305); 1-layer barely passes (0.944), indicating numerical instability in the full 40-layer GGUF Q4_K_M model during autoregressive decode. Cannot be fixed in test code."

# Benchmark added: test_wizardlm_1_0_uncensored_llama2_13b_gguf

## Test
tests/benchmark/test_llms.py::test_wizardlm_1_0_uncensored_llama2_13b_gguf

## Model
- HF name:    TheBloke/WizardLM-1.0-Uncensored-Llama2-13B-GGUF
- Loader:     third_party.tt_forge_models.wizardlm_1_0_uncensored_llama2_13b_gguf.causal_lm.pytorch.loader
- Variant:    ModelVariant.WIZARDLM_1_0_UNCENSORED_LLAMA2_13B_GGUF (13B_GGUF)

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: none (disabled — GGUF Q4_K_M already quantized; bfp_bf8 causes double-quantization with identical catastrophic PCC 0.308)
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, no bfp_bf8)
- Sample per second:  10.9766
- TTFT (ms):          635.868
- Prefill PCC:        0.999490 (PASS)
- First decode PCC:   0.304799 (FAIL — required 0.94)
- Wall clock:         ~13:15
- Hardware:           p150 (blackhole)

## Failure analysis
First decode PCC is catastrophically low (0.305) for the full 40-layer model while
the 1-layer test barely passes (0.944). This pattern indicates numerical instability
that compounds across transformer layers in the autoregressive decode step.

Both configurations tried:
- With bfp_bf8: first_decode PCC = 0.308029 (FAIL)
- Without bfp_bf8: first_decode PCC = 0.304799 (FAIL)

The GGUF Q4_K_M de-quantization (int4 group quantization → fp32) combined with the
TT hardware's numerical processing appears to produce severe quality degradation in
the decode path for the full 40-layer LLaMA-2 13B model. This cannot be fixed in
the test code — it requires either model loader changes or compiler fixes.

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_wizardlm_1_0_uncensored_llama2_13b_gguf_perf_metrics_1.json
Achieved vs top_perf_samples_per_sec: 48.2% (10.977 / 22.780)

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
- total_flops:             822503014528
- breakdown.matmul:        822503014528
- breakdown.linear:        0
- breakdown.conv2d:        0
- breakdown.sparse_matmul: 0

### Inputs
- count:        33
- memory_bytes: 132

### KV cache
- count:        1677721600
- memory_bytes: 3355443200
- memory_gb:    3.125

### Params
- count:                  13015864515
- effective_count:        12852024515
- memory_bytes:           26031729416
- memory_gb:              24.243937261402607
- effective_memory_bytes: 25704049416
- effective_memory_gb:    23.938761480152607
- embedding_count:        163840000
- embedding_memory_bytes: 327680000

### Roofline
- bound:                    dram
- top_perf_samples_per_sec: 22.7802
- top_perf_time_ms:         43.8979
- dram_time_ms:             29.2652
- compute_time_ms_lofi:     0.9347
- compute_time_ms_hifi2:    1.8693
- compute_time_ms_hifi3:    2.8040
- compute_time_ms_hifi4:    3.7387

## Files changed
- tests/benchmark/test_llms.py (added test_wizardlm_1_0_uncensored_llama2_13b_gguf)
- tests/benchmark/benchmarks/llm_benchmark.py (infrastructure fix: graceful handling of missing get_weight_dtype_config_path)
- .github/workflows/perf-bench-matrix.json (added wizardlm_1_0_uncensored_llama2_13b_gguf entry)
- SUMMARY.md

## tt-forge-models submodule
no change
