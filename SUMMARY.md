loader_path: third_party.tt_forge_models.aura_v2_7b_gguf_iq_imatrix.causal_lm.pytorch.loader
variant_id: Aura_v2_7B_GGUF_IQ_Imatrix
arch: p150
status: DONE_PASS
test_function: test_aura_v2_7b_gguf_iq_imatrix
samples_per_second: 34.2
ttft_ms: 296.619276
prefill_pcc: 0.999322
first_decode_pcc: 0.988446
top_perf_samples_per_sec: 44.8551
pct_of_target: 76.2
roofline_bound: dram
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: "bfp_bf8"
failure_reason: null

# Benchmark added: test_aura_v2_7b_gguf_iq_imatrix

## Test
tests/benchmark/test_llms.py::test_aura_v2_7b_gguf_iq_imatrix

## Model
- HF name:    Lewdiculous/Aura_v2_7B-GGUF-IQ-Imatrix
- Loader:     third_party.tt_forge_models.aura_v2_7b_gguf_iq_imatrix.causal_lm.pytorch.loader
- Variant:    ModelVariant.AURA_V2_7B_GGUF_IQ_IMATRIX

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  34.2
- TTFT (ms):          296.619276
- Prefill PCC:        0.999322
- First decode PCC:   0.988446
- Wall clock:         0:09:06
- Hardware:           p150

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_aura_v2_7b_gguf_iq_imatrix_perf_metrics_1.json
Achieved vs top_perf_samples_per_sec: 76.2%

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
- count:                  7241732291
- effective_count:        7110660291
- memory_bytes:           7817470728
- memory_gb:              7.280586965382099
- effective_memory_bytes: 7555326728
- effective_memory_gb:    7.036446340382099
- embedding_count:        131072000
- embedding_memory_bytes: 262144000

### Roofline
- bound:                    dram
- top_perf_samples_per_sec: 44.8551
- top_perf_time_ms:         22.2940
- dram_time_ms:             14.8627
- compute_time_ms_lofi:     0.5171
- compute_time_ms_hifi2:    1.0342
- compute_time_ms_hifi3:    1.5514
- compute_time_ms_hifi4:    2.0685

## Files changed
- tests/benchmark/test_llms.py
- tests/benchmark/benchmarks/llm_benchmark.py (general fix: guard get_weight_dtype_config_path with hasattr)

## tt-forge-models submodule
no change
