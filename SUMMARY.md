loader_path: third_party.tt_forge_models.rinna_japanese_gpt_1b.causal_lm.pytorch.loader
variant_id: 1b
arch: p150
status: DONE_PASS
test_function: test_rinna_japanese_gpt_1b
samples_per_second: 44.369881724845634
ttft_ms: 323.531143
prefill_pcc: 0.999721
first_decode_pcc: 0.999897
top_perf_samples_per_sec: 194.3263
pct_of_target: 22.8
roofline_bound: dram
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: bfp_bf8
failure_reason: null

# Benchmark added: test_rinna_japanese_gpt_1b

## Test
tests/benchmark/test_llms.py::test_rinna_japanese_gpt_1b

## Model
- HF name:    rinna/japanese-gpt-1b
- Loader:     third_party.tt_forge_models.rinna_japanese_gpt_1b.causal_lm.pytorch.loader
- Variant:    ModelVariant.JAPANESE_GPT_1B (value: "1b")

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: bfp_bf8
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  44.37
- TTFT (ms):          323.53
- Prefill PCC:        0.999721
- First decode PCC:   0.999897
- Wall clock:         0:11:40
- Hardware:           p150

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_rinna_japanese_gpt_1b_perf_metrics_1.json
Achieved vs top_perf_samples_per_sec: 22.8% (44.37 / 194.33)

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
- total_flops:             83212369920
- breakdown.matmul:        5888802816
- breakdown.linear:        77323567104
- breakdown.conv2d:        0
- breakdown.sparse_matmul: 0

### Inputs
- count:        33
- memory_bytes: 132

### KV cache
- count:        402653184
- memory_bytes: 805306368
- memory_gb:    0.75

### Params
- count:                  1394725000
- effective_count:        1300615304
- memory_bytes:           1571611156
- memory_gb:              1.4636769481003284
- effective_memory_bytes: 1383391764
- effective_memory_gb:    1.2883839793503284
- embedding_count:        94109696
- embedding_memory_bytes: 188219392

### Roofline
- bound:                    dram
- top_perf_samples_per_sec: 194.3263
- top_perf_time_ms:         5.1460
- dram_time_ms:             3.4307
- compute_time_ms_lofi:     0.0946
- compute_time_ms_hifi2:    0.1891
- compute_time_ms_hifi3:    0.2837
- compute_time_ms_hifi4:    0.3782

## Files changed
- tests/benchmark/test_llms.py (added test_rinna_japanese_gpt_1b)

## tt-forge-models submodule
no change
