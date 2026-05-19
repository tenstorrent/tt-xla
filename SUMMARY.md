loader_path: third_party.tt_forge_models.smol_llama_101m_gqa.causal_lm.pytorch.loader
variant_id: smol_llama_101m_gqa
arch: p150
status: DONE_PASS
test_function: test_smol_llama_101m_gqa
samples_per_second: 518.4113652360423
ttft_ms: 30.799887
prefill_pcc: 0.995367
first_decode_pcc: 0.997321
top_perf_samples_per_sec: 3150.0627
pct_of_target: 16.5
roofline_bound: compute
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: bfp_bf8
failure_reason: null

# Benchmark added: test_smol_llama_101m_gqa

## Test
tests/benchmark/test_llms.py::test_smol_llama_101m_gqa

## Model
- HF name:    BEE-spoke-data/smol_llama-101M-GQA
- Loader:     third_party.tt_forge_models.smol_llama_101m_gqa.causal_lm.pytorch.loader
- Variant:    ModelVariant.SMOL_LLAMA_101M_GQA

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  518.4113652360423
- TTFT (ms):          30.799887
- Prefill PCC:        0.995367
- First decode PCC:   0.997321
- Wall clock:         0:00:55
- Hardware:           p150

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_smol_llama_101m_gqa_perf_metrics_0.json
Achieved vs top_perf_samples_per_sec: 16.5% (518.41 / 3150.06)

Note: The 101M parameter model is very small; the roofline threshold (1e12 FLOPs)
used to distinguish prefill from decode does not separate cleanly for this model.
File _0.json (93B flops, 627 inputs) is identified as the first "decode" graph by
the script. The actual single-token decode graph is _1.json (4.9B flops, 33 inputs,
DRAM-bound, top_perf=3711.82 samples/sec, pct_of_target=13.9%).

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
- total_flops:             93119840864
- breakdown.matmul:        93119840864
- breakdown.linear:        0
- breakdown.conv2d:        0
- breakdown.sparse_matmul: 0

### Inputs
- count:        627
- memory_bytes: 2508

### KV cache
- count:        12582912
- memory_bytes: 25165824
- memory_gb:    0.0234375

### Params
- count:                  101263251
- effective_count:        76588947
- memory_bytes:           130734152
- memory_gb:              0.12175566703081131
- effective_memory_bytes: 81385544
- effective_memory_gb:    0.07579619437456131
- embedding_count:        24674304
- embedding_memory_bytes: 49348608

### Roofline
- bound:                    compute
- top_perf_samples_per_sec: 3150.0627
- top_perf_time_ms:         0.3175
- dram_time_ms:             0.1796
- compute_time_ms_lofi:     0.1058
- compute_time_ms_hifi2:    0.2116
- compute_time_ms_hifi3:    0.3175
- compute_time_ms_hifi4:    0.4233

## Files changed
- tests/benchmark/test_llms.py (added test_smol_llama_101m_gqa)
- tests/benchmark/benchmarks/llm_benchmark.py (general fix: graceful handling of missing get_weight_dtype_config_path method)

## tt-forge-models submodule
no change
