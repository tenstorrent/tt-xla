loader_path: third_party.tt_forge_models.029_shisa_gamma_7b_v1_v2new_dpo405b_i1_gguf.causal_lm.pytorch.loader
variant_id: SHISA_GAMMA_7B_V1_V2NEW_DPO405B_I1_Q4_K_M_GGUF
arch: n150
status: DONE_FAIL
test_function: test_029_shisa_gamma_7b_v1_v2new_dpo405b_i1_gguf
samples_per_second: 18.73
ttft_ms: 641.40
prefill_pcc: 0.998709
first_decode_pcc: 0.554074
top_perf_samples_per_sec: 25.2310
pct_of_target: 74.2
roofline_bound: dram
optimization_level: 2
trace_enabled: null
experimental_weight_dtype: bfp_bf8
failure_reason: "First decode PCC 0.554074 < 0.94; consistent across optimization_level 0/1/2 and trace True/False; bfp_bf8 required (bf16 OOMs); suspected compiler/numerical issue with full-depth GGUF decode graph"

# Benchmark added: test_029_shisa_gamma_7b_v1_v2new_dpo405b_i1_gguf

## Test
tests/benchmark/test_llms.py::test_029_shisa_gamma_7b_v1_v2new_dpo405b_i1_gguf

## Model
- HF name:    mradermacher/029-shisa-gamma-7b-v1-v2new-dpo405b-i1-GGUF
- Loader:     third_party.tt_forge_models.029_shisa_gamma_7b_v1_v2new_dpo405b_i1_gguf.causal_lm.pytorch.loader
- Variant:    SHISA_GAMMA_7B_V1_V2NEW_DPO405B_I1_Q4_K_M_GGUF (Q4_K_M GGUF quantization)

## Test config landed
- optimization_level:        2
- trace_enabled:             true (default)
- experimental_weight_dtype: bfp_bf8 (default; bf16 OOMs — 13.5 GB params)
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  18.73
- TTFT (ms):          641.40
- Prefill PCC:        0.998709 (PASSED)
- First decode PCC:   0.554074 (FAILED — 0.94 required)
- Wall clock:         ~17:34
- Hardware:           n150 (wormhole_b0, single chip, chip_count=2)

## Failure analysis
The test fails on first decode PCC (0.554074 < 0.94) with the full 32-layer model.
Configurations tried:

| optimization_level | trace_enabled | experimental_weight_dtype | Decode PCC | Outcome |
|--------------------|---------------|--------------------------|------------|---------|
| 2 (default)        | true          | bfp_bf8                  | 0.554074   | FAIL    |
| 1                  | true          | bfp_bf8                  | 0.580106   | FAIL    |
| 2                  | false         | bfp_bf8                  | 0.554074   | FAIL    |
| 2                  | true          | ""                        | OOM        | FAIL    |

num_layers=1 passes with PCC=0.999244 across all configs, but the full model consistently fails
decode PCC. Prefill PCC is good (0.998709) in all cases. The decode failure is suspected to be
a compiler or numerical issue specific to the full-depth GGUF decode graph under bfp_bf8 weight
format — not addressable via test configuration changes.

Also fixed: `tests/benchmark/benchmarks/llm_benchmark.py` now uses `hasattr` guard before calling
`model_loader.get_weight_dtype_config_path()`, matching the existing pattern in
`tests/runner/testers/torch/dynamic_torch_model_tester.py`.

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_029_shisa_gamma_7b_v1_v2new_dpo405b_i1_gguf_perf_metrics_1.json
Achieved vs top_perf_samples_per_sec: 74.2% (18.73 / 25.23)

### System
- arch:                        wormhole_b0
- chip_count_in_system_desc:   2
- single_chip_assumption:      True
- worker_grid_cores:           64
- dram_bandwidth_bytes_per_sec: 288000000000

### Peak FLOPs
- lofi:  256000000000000
- hifi2: 128000000000000
- hifi3: 85333333333333
- hifi4: 64000000000000

### Compute
- total_flops:             455065206912
- breakdown.matmul:        455065206912
- breakdown.linear:        0
- breakdown.conv2d:        0
- breakdown.sparse_matmul: 0

### Inputs
- count:        33
- memory_bytes: 132

### KV cache
- count:        268435456
- memory_bytes: 536870912
- memory_gb:    0.5

### Params
- count:                  7241732160
- effective_count:        7110660160
- memory_bytes:           14483464320
- memory_gb:              13.49
- effective_memory_bytes: 14221320320
- effective_memory_gb:    13.24
- embedding_count:        131072000
- embedding_memory_bytes: 262144000

### Roofline
- bound:                    dram
- top_perf_samples_per_sec: 25.2310
- top_perf_time_ms:         39.6338
- dram_time_ms:             26.4225
- compute_time_ms_lofi:     1.7776
- compute_time_ms_hifi2:    3.5552
- compute_time_ms_hifi3:    5.3328
- compute_time_ms_hifi4:    7.1104

## Files changed
- tests/benchmark/test_llms.py — added test_029_shisa_gamma_7b_v1_v2new_dpo405b_i1_gguf (in # FAILED block)
- tests/benchmark/benchmarks/llm_benchmark.py — added hasattr guard for get_weight_dtype_config_path
- SUMMARY.md — this file

## tt-forge-models submodule
no change
