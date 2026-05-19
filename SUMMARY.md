loader_path: third_party.tt_forge_models.codellama_7b_instruct_gguf.causal_lm.pytorch.loader
variant_id: 7B_Instruct_GGUF
arch: p150
status: DONE_PASS
test_function: test_codellama_7b_instruct_gguf
samples_per_second: 3.8134738293734736
ttft_ms: 936.871447
prefill_pcc: 0.999263
first_decode_pcc: 0.982892
top_perf_samples_per_sec: 43.0912
pct_of_target: 8.8
roofline_bound: dram
optimization_level: 1
trace_enabled: true
experimental_weight_dtype: bfp_bf8
failure_reason: null

# Benchmark added: test_codellama_7b_instruct_gguf

## Test
tests/benchmark/test_llms.py::test_codellama_7b_instruct_gguf

## Model
- HF name:    TheBloke/CodeLlama-7B-Instruct-GGUF
- Loader:     third_party.tt_forge_models.codellama_7b_instruct_gguf.causal_lm.pytorch.loader
- Variant:    ModelVariant.CODELLAMA_7B_INSTRUCT_GGUF (7B_Instruct_GGUF)

## Test config landed
- optimization_level:        1
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  3.8134738293734736
- TTFT (ms):          936.871447
- Prefill PCC:        0.999263
- First decode PCC:   0.982892
- Wall clock:         0:26:15
- Hardware:           p150

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_codellama_7b_instruct_gguf_perf_metrics_1.json
Achieved vs top_perf_samples_per_sec: 8.8% (3.813 / 43.09)
Note: optimization_level=2 causes L1 circular buffer clash on p150; capped at level 1

### System
- arch:                        blackhole
- chip_count_in_system_desc:   1
- single_chip_assumption:      True
- worker_grid_cores:           130
- dram_bandwidth_bytes_per_sec: 512000000000

### Peak FLOPs
- lofi:  1040000000000000
- hifi2: 520000000000000
- hifi3: 346666666666666
- hifi4: 260000000000000

### Compute
- total_flops:             425004630144
- breakdown.matmul:        425004630144
- breakdown.linear:        0
- breakdown.conv2d:        0
- breakdown.sparse_matmul: 0

### Inputs
- count:        33
- memory_bytes: 132

### KV cache
- count:        1073741824
- memory_bytes: 2147483648
- memory_gb:    2

### Params
- count:                  6738546886
- effective_count:        6607409350
- memory_bytes:           7282897684
- memory_gb:              6.782727021723986
- effective_memory_bytes: 7020622612
- effective_memory_gb:    6.538464326411486
- embedding_count:        131137536
- embedding_memory_bytes: 262275072

### Roofline
- bound:                    dram
- top_perf_samples_per_sec: 43.0912
- top_perf_time_ms:         23.2066
- dram_time_ms:             15.4711
- compute_time_ms_lofi:     0.4087
- compute_time_ms_hifi2:    0.8173
- compute_time_ms_hifi3:    1.2260
- compute_time_ms_hifi4:    1.6346

## Files changed
- tests/benchmark/test_llms.py

## tt-forge-models submodule
no change
