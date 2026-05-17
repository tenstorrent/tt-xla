loader_path: third_party.tt_forge_models.cerebras_gpt_111m.causal_lm.pytorch.loader
variant_id: base
arch: n150
status: DONE_PASS
test_function: test_cerebras_gpt_111m
samples_per_second: 85.75
ttft_ms: 166.37
prefill_pcc: 0.998543
first_decode_pcc: 0.999136
top_perf_samples_per_sec: 716.8521
pct_of_target: 12.0
roofline_bound: compute
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: bfp_bf8
failure_reason: null

# Benchmark added: test_cerebras_gpt_111m

## Test
tests/benchmark/test_llms.py::test_cerebras_gpt_111m

## Model
- HF name:    cerebras/Cerebras-GPT-111M
- Loader:     third_party.tt_forge_models.cerebras_gpt_111m.causal_lm.pytorch.loader
- Variant:    base

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  85.75
- TTFT (ms):          166.37
- Prefill PCC:        0.998543
- First decode PCC:   0.999136
- Wall clock:         0:03:15
- Hardware:           n150

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_cerebras_gpt_111m_perf_metrics_0.json
Achieved vs top_perf_samples_per_sec: 12.0% (85.75 / 716.85)

Note: Cerebras-GPT-111M is only 111M params. For this tiny model, even the prefill
graph has <1e12 FLOPs (threshold used by extract_perf_targets.py to distinguish
prefill from decode). The script identified file _0 (119G FLOPs, compute-bound)
as the "decode" graph. File _1 (7G FLOPs, dram-bound, top_perf=1079.99) is the
true single-token decode step; the roofline numbers below are from file _0 (prefill).

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
- total_flops:             119038967808
- breakdown.matmul:        41993945088
- breakdown.linear:        77045022720
- breakdown.conv2d:        0
- breakdown.sparse_matmul: 0

### Inputs
- count:        561
- memory_bytes: 2244

### KV cache
- count:        62914560
- memory_bytes: 125829120
- memory_gb:    0.1171875

### Params
- count:                  149648004
- effective_count:        109477764
- memory_bytes:           196894268
- memory_gb:              0.18337207660079002
- effective_memory_bytes: 116553788
- effective_memory_gb:    0.10854917392134666
- embedding_count:        40170240
- embedding_memory_bytes: 80340480

### Roofline
- bound:                    compute
- top_perf_samples_per_sec: 716.8521
- top_perf_time_ms:         1.3950
- dram_time_ms:             0.6173
- compute_time_ms_lofi:     0.4650
- compute_time_ms_hifi2:    0.9300
- compute_time_ms_hifi3:    1.3950
- compute_time_ms_hifi4:    1.8600

## Files changed
- tests/benchmark/test_llms.py (added test_cerebras_gpt_111m)
- tests/benchmark/benchmarks/llm_benchmark.py (general fix: use getattr for get_weight_dtype_config_path)
- .github/workflows/perf-bench-matrix.json (added cerebras_gpt_111m entry)

## tt-forge-models submodule
no change
