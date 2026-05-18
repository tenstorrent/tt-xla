loader_path: third_party.tt_forge_models.argilla_llama.causal_lm.pytorch.loader
variant_id: llama_3_2_1b_instruct_apigen_fc_v0_1
arch: p150
status: DONE_PASS
test_function: test_argilla_llama_3_2_1b_instruct_apigen_fc_v0_1
samples_per_second: 125.163741733176
ttft_ms: 95.901962
prefill_pcc: 0.998444
first_decode_pcc: 0.997606
top_perf_samples_per_sec: 254.0363
pct_of_target: 49.3
roofline_bound: dram
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: "bfp_bf8"
failure_reason: null

# Benchmark added: test_argilla_llama_3_2_1b_instruct_apigen_fc_v0_1

## Test
tests/benchmark/test_llms.py::test_argilla_llama_3_2_1b_instruct_apigen_fc_v0_1

## Model
- HF name:    argilla/Llama-3.2-1B-Instruct-APIGen-FC-v0.1
- Loader:     third_party.tt_forge_models.argilla_llama.causal_lm.pytorch.loader
- Variant:    ModelVariant.LLAMA_3_2_1B_INSTRUCT_APIGEN_FC_V0_1

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  125.163741733176
- TTFT (ms):          95.901962
- Prefill PCC:        0.998444
- First decode PCC:   0.997606
- Wall clock:         0:03:09
- Hardware:           p150

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_argilla_llama_3_2_1b_instruct_apigen_fc_v0_1_perf_metrics_1.json
Achieved vs top_perf_samples_per_sec: 49.3%

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
- total_flops:             79087796288
- breakdown.matmul:        79087796288
- breakdown.linear:        0
- breakdown.conv2d:        0
- breakdown.sparse_matmul: 0

### Inputs
- count:        33
- memory_bytes: 132

### KV cache
- count:        67108864
- memory_bytes: 134217728
- memory_gb:    0.125

### Params
- count:                  1498482851
- effective_count:        1235814563
- memory_bytes:           1838453384
- memory_gb:              1.712193138897419
- effective_memory_bytes: 1313116808
- effective_memory_gb:    1.222935326397419
- embedding_count:        262668288
- embedding_memory_bytes: 525336576

### Roofline
- bound:                    dram
- top_perf_samples_per_sec: 254.0363
- top_perf_time_ms:         3.9364
- dram_time_ms:             2.6243
- compute_time_ms_lofi:     0.0899
- compute_time_ms_hifi2:    0.1797
- compute_time_ms_hifi3:    0.2696
- compute_time_ms_hifi4:    0.3595

## Files changed
- tests/benchmark/test_llms.py
- tests/benchmark/benchmarks/llm_benchmark.py

## tt-forge-models submodule
no change
