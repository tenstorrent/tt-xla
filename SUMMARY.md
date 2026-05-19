loader_path: third_party.tt_forge_models.r1_reward_i1_gguf.causal_lm.pytorch.loader
variant_id: R1_Reward_i1_Q4_K_M_GGUF
arch: p150
status: DONE_FAIL
test_function: test_r1_reward_i1_gguf
samples_per_second: null
ttft_ms: null
prefill_pcc: null
first_decode_pcc: null
top_perf_samples_per_sec: 46.0471
pct_of_target: null
roofline_bound: dram
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: "bfp_bf8"
failure_reason: "g2 (prefill logits graph) TTNN compilation crashes after ~13 min in C++ backend with no Python traceback; g0 and g1 perf graphs compiled and ran successfully; g2 returns extra tensor<32x17x152064xbf16> logits output vs g0/g1; crash is reproducible across multiple runs (runa219, runbd92); diagnostic ttmlir-opt run in progress to capture exact error"

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
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Status: DONE_FAIL
### Reason
The g2 (prefill logits graph) TTNN compilation crashes in the C++ backend after ~13 minutes with
no Python traceback. Crash is reproducible across multiple runs (runa219, runbd92).

Key observations (run started 12:56 UTC May 19, 2026):
- g0 (prefill perf):  compiled 12:57-14:33 (~96 min), ran successfully → perf_metrics_0.json at 14:33
- g1 (decode perf):   compiled 14:35-16:10 (~95 min), ran successfully → perf_metrics_1.json at 16:10
- g2 (prefill logits): TTIR generated at 16:11, TTNN binary compilation CRASHED at 16:23 (~13 min in)
- g3 (decode logits):  never reached

g2 vs g0/g1 difference (from TTIR comparison):
- g0/g1 return: (next_token_ids, next_token_ids_replicated, next_cache_position) — small tensors
- g2 returns extra: tensor<32x17x152064xbf16> — full prefill logits (166 MB), vocab_size=152064

Diagnostic: running ttmlir-opt on the saved g2 TTIR to capture the C++ error message.

Loader issue: The r1_reward_i1_gguf loader's num_layers parameter does NOT effectively truncate
the model. It sets text_config.num_hidden_layers=num_layers but then assigns
`model.model = full_model.model.language_model` (28-layer full LM), overriding the truncation.
This means --num-layers 1 compiles the full 7B model with all 28 layers, making each graph
take ~95 min to compile on p150.

## Measured (full model, defaults)
- Sample per second:  null (test didn't complete PCC phase)
- TTFT (ms):          null
- Prefill PCC:        null
- First decode PCC:   null
- Wall clock:         3:27 (killed before completion)
- Hardware:           p150 (blackhole)

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_r1_reward_i1_gguf_perf_metrics_1.json
Achieved vs top_perf_samples_per_sec: N/A (no measured throughput)

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
- total_flops:             452502433792
- breakdown.matmul:        422903296000
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
- count:                  7615622787
- effective_count:        7070625411
- memory_bytes:           8602865160
- memory_gb:              8.012042529881
- effective_memory_bytes: 7512870408
- effective_memory_gb:    6.9969058111310005
- embedding_count:        544997376
- embedding_memory_bytes: 1089994752

### Roofline
- bound:                    dram
- top_perf_samples_per_sec: 46.0471
- top_perf_time_ms:         21.7169
- dram_time_ms:             14.4779
- compute_time_ms_lofi:     0.5142
- compute_time_ms_hifi2:    1.0284
- compute_time_ms_hifi3:    1.5426
- compute_time_ms_hifi4:    2.0568

## Files changed
- tests/benchmark/test_llms.py (test_r1_reward_i1_gguf added)

## tt-forge-models submodule
no change
