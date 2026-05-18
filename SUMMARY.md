loader_path: third_party.tt_forge_models.capybara_hermes_2_5_mistral_7b_gguf.causal_lm.pytorch.loader
variant_id: CapybaraHermes_2_5_Mistral_7B_GGUF
arch: p150
status: DONE_PASS
test_function: test_capybara_hermes_2_5_mistral_7b_gguf
samples_per_second: 36.022268687577935
ttft_ms: 293.162807
prefill_pcc: 0.999456
first_decode_pcc: 0.993407
top_perf_samples_per_sec: 44.8550
pct_of_target: 80.3
roofline_bound: dram
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: bfp_bf8
failure_reason: null

# Benchmark added: test_capybara_hermes_2_5_mistral_7b_gguf

## Test
tests/benchmark/test_llms.py::test_capybara_hermes_2_5_mistral_7b_gguf

## Model
- HF name:    TheBloke/CapybaraHermes-2.5-Mistral-7B-GGUF
- Loader:     third_party.tt_forge_models.capybara_hermes_2_5_mistral_7b_gguf.causal_lm.pytorch.loader
- Variant:    CapybaraHermes_2_5_Mistral_7B_GGUF

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  36.022268687577935
- TTFT (ms):          293.162807
- Prefill PCC:        0.999456
- First decode PCC:   0.993407
- Wall clock:         0:09:09
- Hardware:           p150

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_capybara_hermes_2_5_mistral_7b_gguf_perf_metrics_1.json
Achieved vs top_perf_samples_per_sec: 80.3% (36.02 / 44.86)

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
- total_flops:             455065731200
- breakdown.matmul:        455065731200
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
- count:                  7241748675
- effective_count:        7110668483
- memory_bytes:           7817495816
- memory_gb:              7.280610330402851
- effective_memory_bytes: 7555335432
- effective_memory_gb:    7.036454446613789
- embedding_count:        131080192
- embedding_memory_bytes: 262160384

### Roofline
- bound:                    dram
- top_perf_samples_per_sec: 44.8550
- top_perf_time_ms:         22.2940
- dram_time_ms:             14.8627
- compute_time_ms_lofi:     0.5171
- compute_time_ms_hifi2:    1.0342
- compute_time_ms_hifi3:    1.5514
- compute_time_ms_hifi4:    2.0685

## Files changed
- tests/benchmark/test_llms.py (added test_capybara_hermes_2_5_mistral_7b_gguf)
- tests/benchmark/benchmarks/llm_benchmark.py (general fix: guard get_weight_dtype_config_path with hasattr)
- .github/workflows/perf-bench-matrix.json (added capybara_hermes_2_5_mistral_7b_gguf entry)

## tt-forge-models submodule
no change
