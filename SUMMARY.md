loader_path: third_party.tt_forge_models.gpt2_wechsel_french.causal_lm.pytorch.loader
variant_id: gpt2_wechsel_french
arch: p150
status: DONE_PASS
test_function: test_gpt2_wechsel_french
samples_per_second: 34.908247562234976
ttft_ms: 151.83362
prefill_pcc: 0.998817
first_decode_pcc: 0.987970
top_perf_samples_per_sec: 1662.0048
pct_of_target: 2.1
roofline_bound: dram
optimization_level: 0
trace_enabled: true
experimental_weight_dtype: null
failure_reason: null

# Benchmark added: test_gpt2_wechsel_french

## Test
tests/benchmark/test_llms.py::test_gpt2_wechsel_french

## Model
- HF name:    benjamin/gpt2-wechsel-french
- Loader:     third_party.tt_forge_models.gpt2_wechsel_french.causal_lm.pytorch.loader
- Variant:    GPT2_WECHSEL_FRENCH

## Test config landed
- optimization_level:        0
- trace_enabled:             true
- experimental_weight_dtype: none
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

Note: optimization_level=2 and optimization_level=1 both fail PCC (0.910 and 0.940
respectively). optimization_level=0 achieves passing PCC of 0.988 with bf16 weights.
experimental_weight_dtype disabled as bfp_bf8 degrades PCC below threshold at all
optimization levels tested.

## Measured (full model, defaults)
- Sample per second:  34.908247562234976
- TTFT (ms):          151.83362
- Prefill PCC:        0.998817
- First decode PCC:   0.987970
- Wall clock:         0:00:51
- Hardware:           p150

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_gpt2_wechsel_french_perf_metrics_0.json
Achieved vs top_perf_samples_per_sec: 2.1%

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
- total_flops:             169253683200
- breakdown.matmul:        55045767168
- breakdown.linear:        114207916032
- breakdown.conv2d:        0
- breakdown.sparse_matmul: 0

### Inputs
- count:        693
- memory_bytes: 2772

### KV cache
- count:        75497472
- memory_bytes: 150994944
- memory_gb:    0.140625

### Params
- count:                  163037319
- effective_count:        123653511
- memory_bytes:           496110104
- memory_gb:              0.4620385393500328
- effective_memory_bytes: 417342488
- effective_memory_gb:    0.38868048042058945
- embedding_count:        39383808
- embedding_memory_bytes: 78767616

### Roofline
- bound:                    dram
- top_perf_samples_per_sec: 1662.0048
- top_perf_time_ms:         0.6017
- dram_time_ms:             0.4011
- compute_time_ms_lofi:     0.1923
- compute_time_ms_hifi2:    0.3847
- compute_time_ms_hifi3:    0.5770
- compute_time_ms_hifi4:    0.7693

## Files changed
- tests/benchmark/test_llms.py (added test_gpt2_wechsel_french)
- tests/benchmark/benchmarks/llm_benchmark.py (general fix: use getattr for get_weight_dtype_config_path)

## tt-forge-models submodule
no change
