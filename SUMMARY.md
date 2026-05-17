loader_path: third_party.tt_forge_models.biomedlm.causal_lm.pytorch.loader
variant_id: BioMedLM
arch: n150
status: DONE_PASS
test_function: test_biomedlm
samples_per_second: 9.762028810948966
ttft_ms: 1048.092965
prefill_pcc: 0.999393
first_decode_pcc: 0.999084
top_perf_samples_per_sec: 57.0635
pct_of_target: 17.1
roofline_bound: dram
optimization_level: 1
trace_enabled: true
experimental_weight_dtype: null
failure_reason: null

# Benchmark added: test_biomedlm

## Test
tests/benchmark/test_llms.py::test_biomedlm

## Model
- HF name:    stanford-crfm/BioMedLM
- Loader:     third_party.tt_forge_models.biomedlm.causal_lm.pytorch.loader
- Variant:    BioMedLM

## Test config landed
- optimization_level:        1
- trace_enabled:             true
- experimental_weight_dtype: none
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

Note: optimization_level=2 causes L1 SpillManagement compiler failure; bfp_bf8 with
opt_level=2 causes PCC=0.867 (below threshold). bfp_bf8 weights without SRAM promotion
(opt_level=1) gives clean PCC but the model default (bfp_bf8) still applies via
DEFAULT_EXPERIMENTAL_WEIGHT_DTYPE. The test omits an explicit experimental_weight_dtype
override, leaving the harness default (bfp_bf8) active.

## Measured (full model, defaults)
- Sample per second:  9.762028810948966
- TTFT (ms):          1048.092965
- Prefill PCC:        0.999393
- First decode PCC:   0.999084
- Wall clock:         0:04:52
- Hardware:           n150

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_biomedlm_perf_metrics_1.json
Achieved vs top_perf_samples_per_sec: 17.1% (9.76 / 57.06)

### System
- arch:                        wormhole_b0
- chip_count_in_system_desc:   2
- single_chip_assumption:      True
- worker_grid_cores:           64
- dram_bandwidth_bytes_per_sec: 288000000000

### Peak FLOPs
- lofi:  256000000000000
- hifi2: 128000000000000
- hifi3: 85333333333333
- hifi4: 64000000000000

### Compute
- total_flops:             165819187200
- breakdown.matmul:        4734320640
- breakdown.linear:        161084866560
- breakdown.conv2d:        0
- breakdown.sparse_matmul: 0

### Inputs
- count:        33
- memory_bytes: 132

### KV cache
- count:        671088640
- memory_bytes: 1342177280
- memory_gb:    1.25

### Params
- count:                  2668221572
- effective_count:        2591626372
- memory_bytes:           2909271564
- memory_gb:              2.70947028324008
- effective_memory_bytes: 2756081164
- effective_memory_gb:    2.566800605505705
- embedding_count:        76595200
- embedding_memory_bytes: 153190400

### Roofline
- bound:                    dram
- top_perf_samples_per_sec: 57.0635
- top_perf_time_ms:         17.5243
- dram_time_ms:             11.6829
- compute_time_ms_lofi:     0.6477
- compute_time_ms_hifi2:    1.2955
- compute_time_ms_hifi3:    1.9432
- compute_time_ms_hifi4:    2.5909

## Files changed
- tests/benchmark/test_llms.py (new test_biomedlm function)
- tests/benchmark/benchmarks/llm_benchmark.py (hasattr guard for get_weight_dtype_config_path)
- .github/workflows/perf-bench-matrix.json (new biomedlm entry)

## tt-forge-models submodule
no change
