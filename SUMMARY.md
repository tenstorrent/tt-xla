loader_path: third_party.tt_forge_models.ace_step_1_5_gguf.causal_lm.pytorch.loader
variant_id: TURBO_GGUF
arch: n150
status: DONE_FAIL
test_function: test_ace_step_1_5_gguf_turbo_gguf
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
failure_reason: "model is ModelTask.MM_CONDITIONAL_GENERATION (DiT decoder for music generation), not a causal LM — incompatible with test_llm harness: ModelLoader has no tokenizer attribute, no generate() method, inputs are continuous tensors (hidden_states/timestep/encoder_hidden_states), not token IDs"

# Benchmark added: test_ace_step_1_5_gguf_turbo_gguf

## Test
tests/benchmark/test_llms.py::test_ace_step_1_5_gguf_turbo_gguf

## Model
- HF name:    ACE-Step/acestep-v15-turbo-shift3
- Loader:     third_party.tt_forge_models.ace_step_1_5_gguf.causal_lm.pytorch.loader
- Variant:    TURBO_GGUF (ModelVariant.ACE_STEP_1_5_TURBO_GGUF)

## Test config landed
- optimization_level:        null (test not runnable)
- trace_enabled:             null (test not runnable)
- experimental_weight_dtype: null (test not runnable)
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Failure Analysis

The ACE-Step 1.5 GGUF model is a **diffusion-based music generation model** using a
DiT (Diffusion Transformer) architecture. Despite residing under the `causal_lm/pytorch/`
directory path in tt-forge-models, its `ModelTask` is `MM_CONDITIONAL_GENERATION`, not
`NLP_CAUSAL_LM`.

The model loader (`ModelLoader`) loads only the DiT decoder sub-model (`full_model.decoder`)
and its `load_inputs()` returns continuous tensor inputs:
- `hidden_states`: `(batch, seq_len, audio_acoustic_hidden_dim)`
- `timestep`: `(batch,)` float
- `timestep_r`: `(batch,)` float
- `attention_mask`: `(batch, seq_len)` long
- `encoder_hidden_states`: `(batch, encoder_seq_len, hidden_size)`
- `encoder_attention_mask`: `(batch, encoder_seq_len)` long
- `context_latents`: `(batch, seq_len, context_dim)`

The `test_llm` harness via `benchmark_llm_torch_xla` immediately calls
`model_loader.tokenizer` (llm_benchmark.py line 80), which raises `AttributeError`
because `ForgeModel` has no tokenizer property and this model has no text generation
capability whatsoever.

Root cause of incompatibility:
1. No tokenizer — model operates on audio latents, not text tokens
2. No `generate()` method — diffusion models do not autoregressively generate text
3. Inputs are continuous tensors — not compatible with the KV-cache prefill/decode harness
4. No logits output — model outputs audio latent predictions, not vocabulary probabilities

This model requires a dedicated diffusion/multimodal benchmark harness, not `test_llm`.

## Measured (full model, defaults)
- Sample per second:  N/A (not run)
- TTFT (ms):          N/A
- Prefill PCC:        N/A
- First decode PCC:   N/A
- Wall clock:         N/A
- Hardware:           n150 (wormhole_b0)

## Decode roofline (first decode graph, single-chip)
Source JSON: N/A (test not run)
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
- tests/benchmark/test_llms.py (added commented-out test_ace_step_1_5_gguf_turbo_gguf stub)
- SUMMARY.md

## tt-forge-models submodule
no change
