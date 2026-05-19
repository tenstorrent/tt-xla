loader_path: third_party.tt_forge_models.rexhaif_qwen3_4b_tulu_sft.causal_lm.pytorch.loader
variant_id: 4B_Tulu_SFT
arch: p150
status: DONE_PASS
test_function: test_rexhaif_qwen3_4b_tulu_sft
samples_per_second: 35.23
ttft_ms: 323.49
prefill_pcc: 0.994876
first_decode_pcc: 0.998415
top_perf_samples_per_sec: 76.54
pct_of_target: 46.0
roofline_bound: dram
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: bfp_bf8
failure_reason: null

# Benchmark added: test_rexhaif_qwen3_4b_tulu_sft

## Test
tests/benchmark/test_llms.py::test_rexhaif_qwen3_4b_tulu_sft

## Model
- HF name:    Rexhaif/Qwen3-4B-Tulu-SFT
- Loader:     third_party.tt_forge_models.rexhaif_qwen3_4b_tulu_sft.causal_lm.pytorch.loader
- Variant:    REXHAIF_QWEN3_4B_TULU_SFT (4B_Tulu_SFT)

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  35.23
- TTFT (ms):          323.49
- Prefill PCC:        0.994876
- First decode PCC:   0.998415
- Wall clock:         0:07:50
- Hardware:           p150

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_rexhaif_qwen3_4b_tulu_sft_perf_metrics_1.json
Achieved vs top_perf_samples_per_sec: 46.0% (35.23 / 76.54)

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
- total_flops:             257425408128
- breakdown.matmul:        257425408128
- breakdown.linear:        0
- breakdown.conv2d:        0
- breakdown.sparse_matmul: 0

### Inputs
- count:        33
- memory_bytes: 132

### KV cache
- count:        301989888
- memory_bytes: 603979776
- memory_gb:    0.5625

### Params
- count:                  4411424451
- effective_count:        4022468291
- memory_bytes:           5051969288
- memory_gb:              4.705013044178486
- effective_memory_bytes: 4274056968
- effective_memory_gb:    3.980525739490986
- embedding_count:        388956160
- embedding_memory_bytes: 777912320

### Roofline
- bound:                    dram
- top_perf_samples_per_sec: 76.5390
- top_perf_time_ms:         13.0652
- dram_time_ms:             8.7102
- compute_time_ms_lofi:     0.2925
- compute_time_ms_hifi2:    0.5851
- compute_time_ms_hifi3:    0.8776
- compute_time_ms_hifi4:    1.1701

## Files changed
- tests/benchmark/test_llms.py (added test_rexhaif_qwen3_4b_tulu_sft)
- tests/benchmark/benchmarks/llm_benchmark.py (general fix: hasattr guard for get_weight_dtype_config_path)
- .github/workflows/perf-bench-matrix.json (added rexhaif_qwen3_4b_tulu_sft entry)
- SUMMARY.md (this file)

## tt-forge-models submodule
93218a34fc → bb809ee68e58fe492169cb89307f3a7aba1d6124
Added Rexhaif/Qwen3-4B-Tulu-SFT causal LM loader (commit 9c1fa66beb) plus subsequent fixes
