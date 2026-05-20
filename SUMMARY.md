loader_path: third_party.tt_forge_models.llama_krikri.causal_lm.pytorch.loader
variant_id: Llama_Krikri_8B_Instruct
arch: p150
status: DONE_PASS
test_function: test_llama_krikri_8b_instruct
samples_per_second: 33.286573210355364
ttft_ms: 307.337858
prefill_pcc: 0.99755
first_decode_pcc: 0.998017
top_perf_samples_per_sec: 42.1142
pct_of_target: 79.0
roofline_bound: dram
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: bfp_bf8
failure_reason: null

# Benchmark added: llama_krikri_8b_instruct

## Test
tests/benchmark/test_llms.py::test_llama_krikri_8b_instruct

## Model
- HF name:    ilsp/Llama-Krikri-8B-Instruct
- Loader:     third_party.tt_forge_models.llama_krikri.causal_lm.pytorch.loader
- Variant:    Llama_Krikri_8B_Instruct

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: bfp_bf8
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  33.286573210355364
- TTFT (ms):          307.337858
- Prefill PCC:        0.997550
- First decode PCC:   0.998017
- Wall clock:         0:08:28
- Hardware:           p150

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_llama_krikri_8b_instruct_perf_metrics_1.json
Achieved vs top_perf_samples_per_sec: 79.0% (33.29 / 42.11)

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
- total_flops:             485801066624
- breakdown.matmul:        485801066624
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
- count:                  8202227907
- effective_count:        7590908099
- memory_bytes:           9288229640
- memory_gb:              8.6503379419446
- effective_memory_bytes: 8065590024
- effective_memory_gb:    7.511666066944599
- embedding_count:        611319808
- embedding_memory_bytes: 1222639616

### Roofline
- bound:                    dram
- top_perf_samples_per_sec: 42.1142
- top_perf_time_ms:         23.7450
- dram_time_ms:             15.8300
- compute_time_ms_lofi:     0.5520
- compute_time_ms_hifi2:    1.1041
- compute_time_ms_hifi3:    1.6561
- compute_time_ms_hifi4:    2.2082

## Files changed
- tests/benchmark/test_llms.py (added test_llama_krikri_8b_instruct)
- tests/benchmark/benchmarks/llm_benchmark.py (fixed missing hasattr guard for get_weight_dtype_config_path)
- .github/workflows/perf-bench-matrix.json (added llama_krikri_8b_instruct entry)

## tt-forge-models submodule
no change
