loader_path: third_party.tt_forge_models.baseline_outcome_reward_qwen_7b_i1_gguf.causal_lm.pytorch.loader
variant_id: 7B_i1_GGUF
arch: p150
status: DONE_FAIL
test_function: test_baseline_outcome_reward_qwen_7b_i1_gguf
samples_per_second: 34.296071599566964
ttft_ms: 257.257439
prefill_pcc: 0.837722
first_decode_pcc: null
top_perf_samples_per_sec: 46.0472
pct_of_target: 74.5
roofline_bound: dram
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: "bfp_bf8"
failure_reason: "7B Q4_K_M GGUF model: accumulated quantization noise across 28 layers yields best Prefill PCC=0.899 (opt_level=0+bfp_bf8), all configs below required 0.94; configs tried: opt_level=0/1/2 × bfp_bf8/no-bfp_bf8, fp32_dest_acc_en=True — none pass PCC threshold on p150"

# Benchmark added: test_baseline_outcome_reward_qwen_7b_i1_gguf

## Test
tests/benchmark/test_llms.py::test_baseline_outcome_reward_qwen_7b_i1_gguf

## Model
- HF name:    mradermacher/Baseline-Outcome-Reward-Qwen-7B-i1-GGUF
- Loader:     third_party.tt_forge_models.baseline_outcome_reward_qwen_7b_i1_gguf.causal_lm.pytorch.loader
- Variant:    ModelVariant.BASELINE_OUTCOME_REWARD_QWEN_7B_I1_GGUF

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model — best PCC attempt: opt_level=0, bfp_bf8)
- Sample per second:  4.151832 (opt_level=0); 34.296072 (opt_level=2, best throughput)
- TTFT (ms):          1243.781843 (opt_level=0); 257.257439 (opt_level=2)
- Prefill PCC:        0.899151 (opt_level=0, bfp_bf8) — best of all configs tried
- First decode PCC:   null (test failed at prefill PCC check)
- Wall clock:         ~4:11 (opt_level=0)
- Hardware:           p150

## PCC investigation summary
| Config | Full-model Prefill PCC |
|---|---|
| opt_level=2, bfp_bf8 | 0.837722 |
| opt_level=2, no bfp_bf8 | 0.784319 |
| opt_level=1, bfp_bf8 | 0.718741 |
| opt_level=1, no bfp_bf8 | 0.848482 |
| opt_level=0, bfp_bf8 | 0.899151 (best) |
| opt_level=0, no bfp_bf8 | 0.892818 |
| opt_level=0, no bfp_bf8+fp32_dest_acc_en | 0.892818 |

Root cause: GGUF Q4_K_M 4-bit quantization introduces weight noise that accumulates
across 28 transformer layers and keeps PCC below the 0.94 threshold on p150.
Infrastructure fix: Added hasattr guard for get_weight_dtype_config_path() in
llm_benchmark.py (general fix for loaders that don't implement this method).

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_baseline_outcome_reward_qwen_7b_i1_gguf_perf_metrics_1.json
Achieved vs top_perf_samples_per_sec: 34.3 / 46.0 = 74.5% (at opt_level=2)

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
- total_flops:             454146588800
- breakdown.matmul:        424547451008
- breakdown.linear:        29599137792
- breakdown.conv2d:        0
- breakdown.sparse_matmul: 0

### Inputs
- count:        33
- memory_bytes: 132

### KV cache
- count:        117440512
- memory_bytes: 234881024
- memory_gb:    0.21875

### Params
- count:                  7615616710
- effective_count:        7070619334
- memory_bytes:           15231233812
- memory_gb:              14.18519188836217
- effective_memory_bytes: 14141239060
- effective_memory_gb:    13.17005516961217
- embedding_count:        544997376
- embedding_memory_bytes: 1089994752

### Roofline
- bound:                    dram
- top_perf_samples_per_sec: 46.0472
- top_perf_time_ms:         21.7169
- dram_time_ms:             14.4779
- compute_time_ms_lofi:     0.5161
- compute_time_ms_hifi2:    1.0322
- compute_time_ms_hifi3:    1.5482
- compute_time_ms_hifi4:    2.0643

## Files changed
- tests/benchmark/test_llms.py (added test_baseline_outcome_reward_qwen_7b_i1_gguf)
- tests/benchmark/benchmarks/llm_benchmark.py (general fix: hasattr guard for get_weight_dtype_config_path)

## tt-forge-models submodule
no change
