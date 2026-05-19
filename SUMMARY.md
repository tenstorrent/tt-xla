loader_path: third_party.tt_forge_models.chronos_gold_12b_gguf.causal_lm.pytorch.loader
variant_id: 12B_1_0_i1_GGUF
arch: p150
status: DONE_PASS
test_function: test_chronos_gold_12b_gguf
samples_per_second: 3.5338915924181014
ttft_ms: 1037.615668
prefill_pcc: 0.995424
first_decode_pcc: 0.997212
top_perf_samples_per_sec: 27.7857
pct_of_target: 12.7
roofline_bound: dram
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: bfp_bf8
failure_reason: null

# Benchmark added: test_chronos_gold_12b_gguf

## Test
tests/benchmark/test_llms.py::test_chronos_gold_12b_gguf

## Model
- HF name:    mradermacher/Chronos-Gold-12B-1.0-i1-GGUF
- Loader:     third_party.tt_forge_models.chronos_gold_12b_gguf.causal_lm.pytorch.loader
- Variant:    ModelVariant.CHRONOS_GOLD_12B_1_0_I1_GGUF

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  3.5338915924181014
- TTFT (ms):          1037.615668
- Prefill PCC:        0.995424
- First decode PCC:   0.997212
- Wall clock:         1:00:00
- Hardware:           p150

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_chronos_gold_12b_gguf_perf_metrics_1.json
Achieved vs top_perf_samples_per_sec: 12.7%

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
- total_flops:             743566213248
- breakdown.matmul:        743566213248
- breakdown.linear:        0
- breakdown.conv2d:        0
- breakdown.sparse_matmul: 0

### Inputs
- count:        33
- memory_bytes: 132

### KV cache
- count:        335544320
- memory_bytes: 671088640
- memory_gb:    0.625

### Params
- count:                  12247782598
- effective_count:        11576693958
- memory_bytes:           13642803988
- memory_gb:              12.705851335078478
- effective_memory_bytes: 12300626708
- effective_memory_gb:    11.455851335078478
- embedding_count:        671088640
- embedding_memory_bytes: 1342177280

### Roofline
- bound:                    dram
- top_perf_samples_per_sec: 27.7857
- top_perf_time_ms:         35.9897
- dram_time_ms:             23.9932
- compute_time_ms_lofi:     0.7150
- compute_time_ms_hifi2:    1.4299
- compute_time_ms_hifi3:    2.1449
- compute_time_ms_hifi4:    2.8599

## Files changed
- tests/benchmark/test_llms.py

## tt-forge-models submodule
no change
