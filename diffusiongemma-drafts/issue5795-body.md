# [vLLM][MRv2] Model Runner v2 migration: feature tracking

Tracks the phased migration of the TT vLLM plugin from the v1 model runner to Model Runner v2 (`model_runner_v2.py`). v2 is the default path for generative models, v1 remains the fallback for anything below that is not yet ported.

MRv2 targets vLLM **0.25.1** as of #5744.

## Phase 1 (completed; #5726)

Core generate path on v2, single- and multi-device, plus the common post-processing features:

- Core generate loop: decode-first multi-pass batch loop, stable-slot request lifecycle (add/update/finish, no condense), TT paged-attention metadata (`prepare_attn`), 2D `[reqs, tokens]` forward layout + `flat_model_io` reshaping
- Separately-compiled forward->logits and sampling graphs (post-processing no longer re-specializes the model forward)
- Sampling: greedy (argmax fast-path), temperature, top-k / top-p, penalties, seeds, on device and on host (`cpu_sampling`)
- Structured output / grammar masking (device and host paths)
- Sample logprobs and prompt logprobs (incl. chunked-prefill partial accumulation)
- LoRA
- Multi-device: tensor parallel, data parallel, 2D mesh, SPMD sharding constraints
- Chunked prefill
- Cross-layer KV sharing
- Text-only inference on multimodal models

#5726 was re-based from 0.22.1 onto post-uplift `main`. The 0.25.1-specific adaptations are: `TTModelState` sets `inputs_embeds_size` itself and matches the new `postprocess_state` signature; `SamplingBatchView` exposes the four attributes the uplifted `from_input_batch` newly reads (`logits_processing_needs_token_ids`, `spec_token_ids`, `logitsprocs`, `logitsprocs_need_output_token_ids`); and the sampler keeps using `allowed_token_ids_additive_mask` after the uplift split that mask into bool + additive variants (a bool mask hits on-device `masked_fill` issues at certain batch x vocab shapes).

## Phase 2 (pending): multimodal / vision enablement

The blocker chain that keeps image models on v1 today. mrope is the root dependency, most vision models need it for interleaved image/text positions.

- M-RoPE (mrope): TT rope-state substitute for `TTModelState` (upstream's `get_rope_state` is Triton-backed and does not run on TT)
- Multimodal image/vision inference end-to-end: encoder runner in `ModelState`, mm profiling / `mm_budget`, on-device validation of the encoder path (`_execute_mm_encoder` / `_gather_mm_embeddings` / `get_mm_embeddings`)
- Un-gate the image tests (`test_gemma4_generation.py`, `test_tensor_parallel_generation.py`) and `test_mrope.py` back onto v2 once the above land

## Phase 3 (pending): remaining features & TT limitations

Large orthogonal features and TT attention/KV-design gaps, not required for the core path:

- Speculative decoding. `num_speculative_steps` is hard-wired to 0 on v2, so v2 now raises `NotImplementedError` on a `speculative_config` rather than silently dropping drafts (v2 is the default runner, so a silent drop would have been a quiet correctness hole). The NGram proposer + multi-step verification landed in #5542 on v1 only and has no e2e test, so nothing regresses today; porting it to v2 needs the draft/ngram proposer plus multi-step verification on the v2 data path.
- Hybrid KV cache with >1 cache type (TT allocator assumes a single cache type per model)
- Encoder-decoder / cross-attention (TT paged-attention metadata is decoder-only)
- KV transfer / disaggregated serving (connector registration)
- `reload_weights`
- Per-tensor weight-dtype overrides (uniform `experimental_weight_dtype` already supported)

## Phase 4 (in progress): non-autoregressive models (DiffusionGemma)

DiffusionGemma (`google/diffusiongemma-26B-A4B-it`) is a discrete diffusion LM: it denoises a fixed 256-token canvas over ~48 iterations rather than emitting one token per step. It runs on one Gemma-4 MoE backbone in two weight-sharing modes: encoder (causal attention, writes KV) and decoder (bidirectional attention, reads fixed KV, never writes). The denoise loop is a host state machine driven by the runner's normal step loop, not an inner loop inside a forward.

vLLM 0.25.1 ships this natively for the GPU runner (`vllm/model_executor/models/diffusion_gemma.py`), so this is a port, not an enable. It needs six runner capabilities that MRv2 does not have yet, each of which is generally useful beyond this one model:

- **Model-specific `ModelState` dispatch.** The runner hard-codes `TTModelState`; the model's own `get_model_state_cls()` returns the upstream GPU state, which cannot run on TT. Needs a TT registry mapping model class -> TT `ModelState` class, defaulting to `TTModelState`.
- **Per-request attention masks.** `prepare_attn` builds a single `TTMetadata` carrying `is_causal: bool` plus one `attn_mask`; bidirectional decode needs a per-request `[num_reqs, 1, S, S]` mask. It must be a float additive mask (0.0 keep / -1e4 masked): a bool mask gives wrong results on the TT SDPA op (device-verified).
- **Span gather over logits.** `_select_hidden_states` gathers only the last token per request; diffusion needs every canvas position, i.e. `[num_decode * canvas_len, vocab]`, and `_sample_from_logits` generalized to `[reqs, canvas_len]`.
- **Read-only / static-prefix KV.** The KV path assumes monotonic append. The decoder must re-read the same fixed encoder KV positions on every denoise step and never append.
- **`custom_sampler()` wiring.** The 0.25.1 `ModelState` ABC has `custom_sampler()` natively, but the TT runner never calls it (`_sample_from_logits` always uses `self.sampler`).
- **Spec-decode data path, overloaded.** The diffusion state machine reuses `num_draft_tokens` / `draft_tokens` / `scheduled_spec_decode_tokens` with diffusion semantics. The structures exist but the runner ignores them, so this overlaps the Phase 3 spec-decode item.

Constraint on validation: the model is Blackhole-only, gated, and ~52 GB, so it cannot run on Wormhole boxes. Full-model validation is CI-only; the port is built as CPU-testable increments in `tests/integrations/vllm_plugin/mrv2/`.

## Out of scope (intentionally stays on v1)

- Pooling / embedding models, different output contract; the worker permanently routes `runner_type == "pooling"` to the v1 runner. Not planned for v2 unless a concrete need arises.
