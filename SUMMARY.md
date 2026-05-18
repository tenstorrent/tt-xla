loader_path: third_party.tt_forge_models.codellama_13b_python_gguf.causal_lm.pytorch.loader
variant_id: CodeLlama_13B_Python_Q4_K_M_GGUF
arch: p150
status: DONE_FAIL
test_function: test_codellama_13b_python_q4_k_m_gguf
samples_per_second: 12.935
ttft_ms: 628.75
prefill_pcc: 0.994976
first_decode_pcc: 0.866785
top_perf_samples_per_sec: 22.7802
pct_of_target: 56.8
roofline_bound: dram
optimization_level: 1
trace_enabled: true
experimental_weight_dtype: bfp_bf8
failure_reason: "First decode PCC consistently fails (0.83-0.87) across all opt levels (0,1,2), with/without bfp_bf8, with/without fp32_dest_acc_en; prefill PCC consistently good (>0.994). Suspected KV-cache numerical precision accumulation over 40 layers with Q4_K_M GGUF dequantized to bfloat16."

# Benchmark added: test_codellama_13b_python_q4_k_m_gguf

## Test
tests/benchmark/test_llms.py::test_codellama_13b_python_q4_k_m_gguf

## Model
- HF name:    TheBloke/CodeLlama-13B-Python-GGUF
- Loader:     third_party.tt_forge_models.codellama_13b_python_gguf.causal_lm.pytorch.loader
- Variant:    CodeLlama_13B_Python_Q4_K_M_GGUF

## Test config landed
- optimization_level:        2 (DEFAULT_OPTIMIZATION_LEVEL — not hard-coded, test uses default)
- trace_enabled:             true (DEFAULT — not hard-coded)
- experimental_weight_dtype: "bfp_bf8" (DEFAULT — not hard-coded)
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, best passing config — all configurations FAIL PCC)
Best config tested: optimization_level=1, trace_enabled=True, experimental_weight_dtype=bfp_bf8
- Sample per second:  12.935 (decode throughput)
- TTFT (ms):          628.75
- Prefill PCC:        0.994976 (PASSED)
- First decode PCC:   0.866785 (FAILED — required 0.94)
- Hardware:           p150

## PCC Failure Summary
All configurations tested fail First Decode PCC; Prefill PCC always passes:

| Configuration                                  | Prefill PCC | Decode PCC | Status  |
|------------------------------------------------|-------------|------------|---------|
| opt_level=2, bfp_bf8=True, trace=True          | 0.996230    | 0.833985   | FAILED  |
| opt_level=2, bfp_bf8=False, trace=True         | 0.997897    | 0.832305   | FAILED  |
| opt_level=1, bfp_bf8=True, trace=True          | 0.994976    | 0.866785   | FAILED  |
| opt_level=1, fp32_dest_acc_en=True             | 0.994976    | 0.866785   | FAILED  |
| opt_level=0, bfp_bf8=False, trace=False        | 0.997366    | 0.841102   | FAILED  |

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_codellama_13b_python_q4_k_m_gguf_perf_metrics_1.json
Achieved vs top_perf_samples_per_sec: 56.8% (12.935 / 22.78)

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
- count:                  13015864384
- effective_count:        12852024384
- memory_bytes:           26031728768
- memory_gb:              24.24
- effective_memory_bytes: 25704048768
- effective_memory_gb:    23.94
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
- tests/benchmark/test_llms.py (test already existed; no changes made)
- .github/workflows/perf-bench-matrix.json (added codellama_13b_python_q4_k_m_gguf entry)

## tt-forge-models submodule
no change
