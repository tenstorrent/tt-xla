loader_path: third_party.tt_forge_models.smollm2_360m_instruct_gguf.causal_lm.pytorch.loader
variant_id: Q8_0
arch: p150
status: DONE_PASS
test_function: test_smollm2_360m_instruct_gguf
samples_per_second: 129.06
ttft_ms: 148.14
prefill_pcc: 0.995379
first_decode_pcc: 0.996508
top_perf_samples_per_sec: 4728.6082
pct_of_target: 2.7
roofline_bound: compute
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: bfp_bf8
failure_reason: null

# Benchmark added: smollm2_360m_instruct_gguf

## Test
tests/benchmark/test_llms.py::test_smollm2_360m_instruct_gguf

## Model
- HF name:    bartowski/SmolLM2-360M-Instruct-GGUF
- Loader:     third_party.tt_forge_models.smollm2_360m_instruct_gguf.causal_lm.pytorch.loader
- Variant:    ModelVariant.SMOLLM2_360M_INSTRUCT_Q8_0

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  129.06
- TTFT (ms):          148.14
- Prefill PCC:        0.995379
- First decode PCC:   0.996508
- Wall clock:         0:04:39
- Hardware:           p150

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_smollm2_360m_instruct_gguf_perf_metrics_0.json
Achieved vs top_perf_samples_per_sec: 2.7%

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
- total_flops:             62033757248
- breakdown.matmul:        62033757248
- breakdown.linear:        0
- breakdown.conv2d:        0
- breakdown.sparse_matmul: 0

### Inputs
- count:        561
- memory_bytes: 2244

### KV cache
- count:        2621440
- memory_bytes: 5242880
- memory_gb:    0.0048828125

### Params
- count:                  104205283
- effective_count:        57019363
- memory_bytes:           154958088
- memory_gb:              0.14431596547365189
- effective_memory_bytes: 60586248
- effective_memory_gb:    0.056425340473651886
- embedding_count:        47185920
- embedding_memory_bytes: 94371840

### Roofline
- bound:                    compute
- top_perf_samples_per_sec: 4728.6082
- top_perf_time_ms:         0.2115
- dram_time_ms:             0.1201
- compute_time_ms_lofi:     0.0705
- compute_time_ms_hifi2:    0.1410
- compute_time_ms_hifi3:    0.2115
- compute_time_ms_hifi4:    0.2820

## Files changed
- tests/benchmark/test_llms.py
- tests/benchmark/benchmarks/llm_benchmark.py
- .github/workflows/perf-bench-matrix.json

## tt-forge-models submodule
no change
