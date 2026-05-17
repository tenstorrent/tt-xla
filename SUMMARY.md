loader_path: third_party.tt_forge_models.baseline_outcome_reward_qwen_7b_i1_gguf.causal_lm.pytorch.loader
variant_id: 7B_i1_GGUF
arch: n150
status: DONE_FAIL
test_function: test_baseline_outcome_reward_qwen_7b_i1_gguf
samples_per_second: 18.9561
ttft_ms: 606.868
prefill_pcc: 0.872029
first_decode_pcc: null
top_perf_samples_per_sec: 25.9015
pct_of_target: 73.2
roofline_bound: dram
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: "bfp_bf8"
failure_reason: "bfp_bf8 required to fit 7B model in DRAM (~14GB at bf16) but causes Prefill PCC=0.872 (opt_level=2) / 0.890 (opt_level=1) below required 0.94; without bfp_bf8 hits DRAM OOM"

# Benchmark added: test_baseline_outcome_reward_qwen_7b_i1_gguf

## Test
tests/benchmark/test_llms.py::test_baseline_outcome_reward_qwen_7b_i1_gguf

## Model
- HF name:    mradermacher/Baseline-Outcome-Reward-Qwen-7B-i1-GGUF
- Loader:     third_party.tt_forge_models.baseline_outcome_reward_qwen_7b_i1_gguf.causal_lm.pytorch.loader
- Variant:    7B_i1_GGUF

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8" (default — required to fit in DRAM, but causes PCC failure)
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, optimization_level=2 with bfp_bf8)
- Sample per second:  18.9561 (PCC failed; numbers from failed run)
- TTFT (ms):          606.868
- Prefill PCC:        0.872029 (FAILED — threshold 0.94)
- First decode PCC:   N/A (test failed at prefill PCC assertion)
- Wall clock:         ~10 min
- Hardware:           n150 (wormhole_b0, n300 board single-chip assumption)

## Failure summary
- With bfp_bf8 + optimization_level=2: Prefill PCC=0.872 < 0.94 required
- With bfp_bf8 + optimization_level=1: Prefill PCC=0.890 < 0.94 required
- Without bfp_bf8 + optimization_level=2: DRAM OOM (model ~14GB at bf16, device DRAM insufficient)
- Root cause: 7B GGUF model (Q4_K_M quantized) is dequantized to bf16 by transformers, requiring bfp_bf8
  weight compression to fit in DRAM, but that additional quantization error accumulates over 28 layers
  and pushes prefill PCC below the required threshold.

## Infrastructure fix
- Fixed a general bug in tests/benchmark/benchmarks/llm_benchmark.py where
  `model_loader.get_weight_dtype_config_path()` was called unconditionally without
  checking `hasattr`. Now guarded with `if hasattr(model_loader, "get_weight_dtype_config_path")`.
  The runner (`tests/runner/testers/torch/dynamic_torch_model_tester.py`) already did this check.

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_baseline_outcome_reward_qwen_7b_i1_gguf_perf_metrics_1.json
Achieved vs top_perf_samples_per_sec: 73.2% (18.9561 / 25.9015)

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
- total_flops:             452502421632
- breakdown.matmul:        422903283840
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
- count:                  7615616707
- effective_count:        7070619331
- memory_bytes:           8602840840
- memory_gb:              8.012019880115986
- effective_memory_bytes: 7512846088
- effective_memory_gb:    6.996883161365986
- embedding_count:        544997376
- embedding_memory_bytes: 1089994752

### Roofline
- bound:                    dram
- top_perf_samples_per_sec: 25.9015
- top_perf_time_ms:         38.6077
- dram_time_ms:             25.7385
- compute_time_ms_lofi:     1.7676
- compute_time_ms_hifi2:    3.5352
- compute_time_ms_hifi3:    5.3028
- compute_time_ms_hifi4:    7.0704

## Files changed
- tests/benchmark/test_llms.py (added test_baseline_outcome_reward_qwen_7b_i1_gguf with # FAILED comment)
- tests/benchmark/benchmarks/llm_benchmark.py (infrastructure fix: guard get_weight_dtype_config_path with hasattr)

## tt-forge-models submodule
no change
