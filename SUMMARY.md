loader_path: third_party.tt_forge_models.lorablated_w2bb_psy_della_i1_gguf.causal_lm.pytorch.loader
variant_id: i1_GGUF
arch: p150
status: DONE_PASS
test_function: test_lorablated_w2bb_psy_della_i1_gguf
samples_per_second: 21.012014480386213
ttft_ms: 471.80256
prefill_pcc: 0.995916
first_decode_pcc: 0.998074
top_perf_samples_per_sec: 27.7857
pct_of_target: 75.6
roofline_bound: dram
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: bfp_bf8
failure_reason: null

# Benchmark added: test_lorablated_w2bb_psy_della_i1_gguf

## Test
tests/benchmark/test_llms.py::test_lorablated_w2bb_psy_della_i1_gguf

## Model
- HF name:    mradermacher/Lorablated-w2bb-psy-della-i1-GGUF
- Loader:     third_party.tt_forge_models.lorablated_w2bb_psy_della_i1_gguf.causal_lm.pytorch.loader
- Variant:    ModelVariant.LORABLATED_W2BB_PSY_DELLA_I1_GGUF

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: bfp_bf8
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  21.012014480386213
- TTFT (ms):          471.80256
- Prefill PCC:        0.995916
- First decode PCC:   0.998074
- Wall clock:         0:26:04
- Hardware:           p150

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_lorablated_w2bb_psy_della_i1_gguf_perf_metrics_1.json
Achieved vs top_perf_samples_per_sec: 75.6%

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
- total_flops:             740882514048
- breakdown.matmul:        740882514048
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
- count:                  12247803075
- effective_count:        11576704195
- memory_bytes:           13642835336
- memory_gb:              12.705880530178547
- effective_memory_bytes: 12300637576
- effective_memory_gb:    11.455861456692219
- embedding_count:        671098880
- embedding_memory_bytes: 1342197760

### Roofline
- bound:                    dram
- top_perf_samples_per_sec: 27.7857
- top_perf_time_ms:         35.9898
- dram_time_ms:             23.9932
- compute_time_ms_lofi:     0.8419
- compute_time_ms_hifi2:    1.6838
- compute_time_ms_hifi3:    2.5257
- compute_time_ms_hifi4:    3.3676

## Files changed
- tests/benchmark/test_llms.py
- .github/workflows/perf-bench-matrix.json

## tt-forge-models submodule
no change — submodule remains at 917494c886
