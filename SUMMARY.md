loader_path: third_party.tt_forge_models.r1_reward_i1_gguf.causal_lm.pytorch.loader
variant_id: R1_Reward_i1_Q4_K_M_GGUF
arch: p150
status: DONE_PASS
test_function: test_r1_reward_i1_gguf
samples_per_second: 4.188103573322335
ttft_ms: 1242.746089
prefill_pcc: 0.992480
first_decode_pcc: 0.996705
top_perf_samples_per_sec: 46.0471
pct_of_target: 9.1
roofline_bound: dram
optimization_level: 0
trace_enabled: true
experimental_weight_dtype: "bfp_bf8"
failure_reason: null

# Benchmark added: test_r1_reward_i1_gguf

## Test
tests/benchmark/test_llms.py::test_r1_reward_i1_gguf

## Model
- HF name:    yifanzhang114/R1-Reward (mradermacher/R1-Reward-i1-GGUF Q4_K_M)
- Loader:     third_party.tt_forge_models.r1_reward_i1_gguf.causal_lm.pytorch.loader
- Variant:    ModelVariant.R1_REWARD_I1_Q4_K_M_GGUF
- Params:     ~7.6B (Qwen2.5-7B causal LM extracted from Qwen2_5_VLForConditionalGeneration)
- Architecture: Qwen2ForCausalLM (extracted from full VL model)

## Test config landed
- optimization_level:        0
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Status: DONE_PASS

### Run summary
Full-model run on p150 (blackhole), optimization_level=0, completed in 7m38s.

Key observations:
- At optimization_level=1 and 2: PJRT device optimizer (`getOrCreateOptimizerSubmesh`)
  opens a real device connection for each graph. After g0+g1 compile (~95 min each),
  device state becomes corrupted and g2 crashes ~13 min in with no Python traceback.
  Confirmed via ttmlir-opt: g2 TTIR compiles cleanly past 13 min without the device
  optimizer → crash is in PJRT device state, not the TTNN compiler itself.
- At optimization_level=0: No device optimizer opened. All 4 graphs compile in ~7 min
  total (vs ~195 min at opt_level=2). PCC passes on both prefill and decode.
- Low pct_of_target (9.1%) is expected at opt_level=0: without SRAM tensor placement,
  all weight reads are DRAM → actual decode latency is ~239ms vs 21.7ms theoretical.

## Measured (full model, optimization_level=0)
- Sample per second:  4.188103573322335
- TTFT (ms):          1242.746089
- Prefill PCC:        0.992480
- First decode PCC:   0.996705
- Wall clock:         7:38 (458.20s)
- Hardware:           p150 (blackhole)

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_r1_reward_i1_gguf_perf_metrics_1.json

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
- total_flops:             454146600960
- breakdown.matmul:        424547463168
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
- count:                  7615622790
- effective_count:        7070625414
- memory_bytes:           8602865172
- memory_gb:              8.012042541056871
- effective_memory_bytes: 7512870420
- effective_memory_gb:    6.996905822306871
- embedding_count:        544997376
- embedding_memory_bytes: 1089994752

### Roofline
- bound:                    dram
- top_perf_samples_per_sec: 46.0471
- top_perf_time_ms:         21.7169
- dram_time_ms:             14.4779
- compute_time_ms_lofi:     0.5161
- compute_time_ms_hifi2:    1.0322
- compute_time_ms_hifi3:    1.5482
- compute_time_ms_hifi4:    2.0643

## Files changed
- tests/benchmark/test_llms.py (test_r1_reward_i1_gguf added, optimization_level=0)
- tests/benchmark/benchmarks/llm_benchmark.py (hasattr guard for get_weight_dtype_config_path)

## tt-forge-models submodule
no change
