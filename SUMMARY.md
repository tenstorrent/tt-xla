loader_path: third_party.tt_forge_models.apriel.causal_lm.pytorch.loader
variant_id: 5B_Instruct
arch: n150
status: DONE_FAIL
test_function: test_apriel_5b_instruct
samples_per_second: null
ttft_ms: null
prefill_pcc: null
first_decode_pcc: null
top_perf_samples_per_sec: 73.0649
pct_of_target: null
roofline_bound: dram
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: bfp_bf8
failure_reason: "Prefill PCC consistently fails below 0.94 (range: 0.23-0.43) on full 28-layer model across all configurations tested (optimization_level=0/2, bfp_bf8, fp32_dest_acc_en=True). Custom Apriel YaRN RoPE architecture (model_type='apriel', rope_type='yarn', factor=32.0) causes accumulating numerical errors across layers that the tt-mlir compiler does not handle correctly. Infrastructure fix applied: added hasattr guard for get_weight_dtype_config_path in llm_benchmark.py (general fix for loaders without this method)."

# Benchmark added: test_apriel_5b_instruct

## Test
tests/benchmark/test_llms.py::test_apriel_5b_instruct

## Model
- HF name:    ServiceNow-AI/Apriel-5B-Instruct
- Loader:     third_party.tt_forge_models.apriel.causal_lm.pytorch.loader
- Variant:    ModelVariant.APRIEL_5B_INSTRUCT (= "5B_Instruct")

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: bfp_bf8
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  N/A (test fails PCC)
- TTFT (ms):          N/A
- Prefill PCC:        N/A — consistently 0.23-0.43 (fails 0.94 threshold)
- First decode PCC:   N/A
- Wall clock:         N/A
- Hardware:           n150

## Bring-up notes
The model loaded and compiled successfully. The benchmark infrastructure
required a general fix: `llm_benchmark.py` line 480 called
`model_loader.get_weight_dtype_config_path()` unconditionally, but the
Apriel loader (and potentially other loaders) does not have this method.
The fix adds a `hasattr` guard matching the existing pattern in
`tests/runner/testers/torch/dynamic_torch_model_tester.py`.

PCC failures observed across all tested configurations:

| Configuration               | Layers | Prefill PCC | Result |
|-----------------------------|--------|-------------|--------|
| opt=2, bfp_bf8 (default)    | Full   | 0.374       | FAIL   |
| opt=2, bfp_bf8 (default)    | Full   | 0.320       | FAIL   |
| opt=2, bfp_bf8 (fresh cache)| Full   | 0.234       | FAIL   |
| opt=2, no bfp               | Full   | 0.433       | FAIL   |
| fp32_dest_acc_en=True, opt=2| Full   | 0.345       | FAIL   |
| fp32_dest_acc_en=True, opt=0| Full   | 0.305       | FAIL   |
| fp32_dest_acc_en=True, opt=2| 1 layer| 0.992      | PASS   |

The single-layer model passes PCC with fp32_dest_acc_en=True, but errors
accumulate over the full 28 layers. The warning "Device prefill produced
different tokens than CPU prefill" appears in all full-model runs, indicating
the attention mechanism is computing fundamentally different results on device.

Root cause: The Apriel model uses a custom architecture (`model_type='apriel'`,
`trust_remote_code=True`) with YaRN RoPE scaling (factor=32.0, rope_theta=1e6).
The custom `modeling_apriel.py` contains operations that the tt-mlir compiler
does not lower accurately across all 28 layers.

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_apriel_5b_instruct_perf_metrics_1.json
Achieved vs top_perf_samples_per_sec: N/A (test fails PCC)

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
- total_flops:             274877907072
- breakdown.matmul:        274877907072
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
- count:                  4832071875
- effective_count:        4295200963
- memory_bytes:           5637612298
- memory_gb:              5.250435600057244
- effective_memory_bytes: 4563870474
- effective_memory_gb:    4.250435600057244
- embedding_count:        536870912
- embedding_memory_bytes: 1073741824

### Roofline
- bound:                    dram
- top_perf_samples_per_sec: 73.0649
- top_perf_time_ms:         13.6865
- dram_time_ms:             9.1243
- compute_time_ms_lofi:     0.3124
- compute_time_ms_hifi2:    0.6247
- compute_time_ms_hifi3:    0.9371
- compute_time_ms_hifi4:    1.2494

## Files changed
- tests/benchmark/test_llms.py (added test_apriel_5b_instruct)
- tests/benchmark/benchmarks/llm_benchmark.py (general fix: hasattr guard for get_weight_dtype_config_path)
- .github/workflows/perf-bench-matrix.json (added apriel_5b_instruct entry)
- SUMMARY.md

## tt-forge-models submodule
no change
