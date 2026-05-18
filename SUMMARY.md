loader_path: third_party.tt_forge_models.gemma_2_2b_it_q4f16_0_mlc.causal_lm.pytorch.loader
variant_id: gemma_2_2b_it_q4f16_0_mlc
arch: p150
status: DONE_PASS
test_function: test_gemma_2_2b_it_q4f16_0_mlc
samples_per_second: 55.75383147032995
ttft_ms: 259.926714
prefill_pcc: 0.999489
first_decode_pcc: 0.998022
top_perf_samples_per_sec: 116.8566
pct_of_target: 47.7
roofline_bound: dram
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: bfp_bf8
failure_reason: null

# Benchmark added: test_gemma_2_2b_it_q4f16_0_mlc

## Test
tests/benchmark/test_llms.py::test_gemma_2_2b_it_q4f16_0_mlc

## Model
- HF name:    unsloth/gemma-2-2b-it
- Loader:     third_party.tt_forge_models.gemma_2_2b_it_q4f16_0_mlc.causal_lm.pytorch.loader
- Variant:    GEMMA_2_2B_IT_Q4F16_0_MLC

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  55.75383147032995
- TTFT (ms):          259.926714
- Prefill PCC:        0.999489
- First decode PCC:   0.998022
- Wall clock:         0:07:56
- Hardware:           p150

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_gemma_2_2b_it_q4f16_0_mlc_perf_metrics_2.json
Achieved vs top_perf_samples_per_sec: 47.7%

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
- total_flops:             167302398208
- breakdown.matmul:        167302398208
- breakdown.linear:        0
- breakdown.conv2d:        0
- breakdown.sparse_matmul: 0

### Inputs
- count:        34
- memory_bytes: 134

### KV cache
- count:        218103808
- memory_bytes: 436207616
- memory_gb:    0.40625

### Params
- count:                  3204166151
- effective_count:        2614342151
- memory_bytes:           3958097940
- memory_gb:              3.686265964061022
- effective_memory_bytes: 2778449940
- effective_memory_gb:    2.587633151561022
- embedding_count:        589824000
- embedding_memory_bytes: 1179648000

### Roofline
- bound:                    dram
- top_perf_samples_per_sec: 116.8566
- top_perf_time_ms:         8.5575
- dram_time_ms:             5.7050
- compute_time_ms_lofi:     0.1901
- compute_time_ms_hifi2:    0.3802
- compute_time_ms_hifi3:    0.5703
- compute_time_ms_hifi4:    0.7605

## Files changed
- tests/benchmark/test_llms.py
- tests/benchmark/benchmarks/llm_benchmark.py (general fix: guard get_weight_dtype_config_path with hasattr check)
- .github/workflows/perf-bench-matrix.json

## tt-forge-models submodule
no change
