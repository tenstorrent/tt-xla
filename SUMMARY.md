loader_path: third_party.tt_forge_models.captain_eris_violet_gguf.causal_lm.pytorch.loader
variant_id: quantfactory_V0.420_12B_GGUF
arch: p150
status: DONE_PASS
test_function: test_captain_eris_violet_gguf_12b
samples_per_second: 20.64
ttft_ms: 477.4
prefill_pcc: 0.997689
first_decode_pcc: 0.996325
top_perf_samples_per_sec: 27.7857
pct_of_target: 74.3
roofline_bound: dram
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: bfp_bf8
failure_reason: null

# Benchmark added: test_captain_eris_violet_gguf_12b

## Test
tests/benchmark/test_llms.py::test_captain_eris_violet_gguf_12b

## Model
- HF name:    QuantFactory/Captain-Eris_Violet-V0.420-12B-GGUF
- Loader:     third_party.tt_forge_models.captain_eris_violet_gguf.causal_lm.pytorch.loader
- Variant:    ModelVariant.QUANTFACTORY_CAPTAIN_ERIS_VIOLET_V0_420_12B_GGUF

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  20.64
- TTFT (ms):          477.4
- Prefill PCC:        0.997689
- First decode PCC:   0.996325
- Wall clock:         0:13:22
- Hardware:           p150

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_captain_eris_violet_gguf_12b_perf_metrics_1.json
Achieved vs top_perf_samples_per_sec: 74.3% (20.64 / 27.79)

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
- total_flops:             740881858688
- breakdown.matmul:        740881858688
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
- count:                  12247782595
- effective_count:        11576693955
- memory_bytes:           13642803976
- memory_gb:              12.706
- effective_memory_bytes: 12300626696
- effective_memory_gb:    11.456
- embedding_count:        671088640
- embedding_memory_bytes: 1342177280

### Roofline
- bound:                    dram
- top_perf_samples_per_sec: 27.7857
- top_perf_time_ms:         35.9897
- dram_time_ms:             23.9932
- compute_time_ms_lofi:     0.8419
- compute_time_ms_hifi2:    1.6838
- compute_time_ms_hifi3:    2.5257
- compute_time_ms_hifi4:    3.3676

## Files changed
- tests/benchmark/test_llms.py (added test_captain_eris_violet_gguf_12b)
- tests/benchmark/benchmarks/llm_benchmark.py (infrastructure fix: guard get_weight_dtype_config_path with hasattr)

## tt-forge-models submodule
no change
