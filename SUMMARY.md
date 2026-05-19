loader_path: third_party.tt_forge_models.mistral_nemo_instruct_thinking_gguf.causal_lm.pytorch.loader
variant_id: 12B_Thinking_i1_GGUF
arch: n150
status: DONE_FAIL
test_function: test_mistral_nemo_instruct_thinking_gguf
samples_per_second: null
ttft_ms: null
prefill_pcc: null
first_decode_pcc: null
top_perf_samples_per_sec: null
pct_of_target: null
roofline_bound: null
optimization_level: null
trace_enabled: null
experimental_weight_dtype: null
failure_reason: "model size ~12B exceeds 10B single-chip capacity on n150"

# Benchmark added: mistral_nemo_instruct_thinking_gguf

## Test
tests/benchmark/test_llms.py::test_mistral_nemo_instruct_thinking_gguf

## Model
- HF name:    mradermacher/Mistral-Nemo-Instruct-2407-12B-Thinking-M-Claude-Opus-High-Reasoning-i1-GGUF
- Loader:     third_party.tt_forge_models.mistral_nemo_instruct_thinking_gguf.causal_lm.pytorch.loader
- Variant:    12B_Thinking_i1_GGUF

## Test config landed
- optimization_level:        N/A (early exit — size exceeds single-chip cap)
- trace_enabled:             N/A
- experimental_weight_dtype: N/A
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  N/A
- TTFT (ms):          N/A
- Prefill PCC:        N/A
- First decode PCC:   N/A
- Wall clock:         N/A
- Hardware:           n150

## Reason for early exit
The variant `12B_Thinking_i1_GGUF` has a parameter count of ~12B, which exceeds
the 10B single-chip capacity ceiling for n150 (wormhole_b0). The skill's Step 1.6
rule rejects models with parameter size token > 10B on n150 to avoid burning a
device-hour on a guaranteed OOM during weight transfer.

The model is a GGUF Q4_K_M quantized variant (~6 GB effective weight size at 4-bit),
but the parameter-count-based cap (12B > 10B) applies regardless of quantization.

If this model should run on n150 with GGUF quantization (because the effective
weight memory fits within DRAM), update the skill's single-chip cap rule and
re-queue. Alternatively, route to a p150 worker (cap: 25B) where this model
fits comfortably.

An existing tensor-parallel variant is already tested at:
  tests/benchmark/test_llms.py::test_mistral_nemo_instruct_2407_tp

## Decode roofline (first decode graph, single-chip)
Source JSON: N/A
Achieved vs top_perf_samples_per_sec: N/A

### System
- arch:                        n150
- chip_count_in_system_desc:   N/A
- single_chip_assumption:      N/A
- worker_grid_cores:           N/A
- dram_bandwidth_bytes_per_sec: N/A

### Peak FLOPs
- lofi:  N/A
- hifi2: N/A
- hifi3: N/A
- hifi4: N/A

### Compute
- total_flops:             N/A
- breakdown.matmul:        N/A
- breakdown.linear:        N/A
- breakdown.conv2d:        N/A
- breakdown.sparse_matmul: N/A

### Inputs
- count:        N/A
- memory_bytes: N/A

### KV cache
- count:        N/A
- memory_bytes: N/A
- memory_gb:    N/A

### Params
- count:                  N/A
- effective_count:        N/A
- memory_bytes:           N/A
- memory_gb:              N/A
- effective_memory_bytes: N/A
- effective_memory_gb:    N/A
- embedding_count:        N/A
- embedding_memory_bytes: N/A

### Roofline
- bound:                    N/A
- top_perf_samples_per_sec: N/A
- top_perf_time_ms:         N/A
- dram_time_ms:             N/A
- compute_time_ms_lofi:     N/A
- compute_time_ms_hifi2:    N/A
- compute_time_ms_hifi3:    N/A
- compute_time_ms_hifi4:    N/A

## Files changed
- SUMMARY.md (added)

## tt-forge-models submodule
no change
