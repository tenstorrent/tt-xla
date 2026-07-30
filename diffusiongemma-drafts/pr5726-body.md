## Transition to vLLM Model Runner v2 (MRv2) for the TT plugin

The old monolithic `TTModelRunner` (a fork of upstream's TPU runner) is replaced by MRv2's four piece split: runner (`TTModelRunnerV2`) plus `TTModelState` plus `TTRequestState` plus `TTInputBatch`, forked and adapted for TT (upstream's v2 is Triton/CUDA gated, so we cannot subclass `GPUModelRunner`). Parallelism uses `mark_sharding`/GSPMD (no explicit collectives).

### Base

Rebased onto `main` at vLLM **0.25.1** (post #5744). This PR previously targeted 0.22.1; the branch was force-pushed onto the post-uplift main, so earlier inline review comments may show as outdated. Five commits:

1. Forward-port MRv2 runner onto vLLM 0.25.1
2. Fix logger initialization in MRv2
3. Split forward/sampling graphs; pin mrope + image tests to v1
4. Adapt v2 sampling bridge to the uplifted `from_input_batch`
5. Reject `speculative_config` on the v2 runner

### Default runner
`TTConfig.use_v2_model_runner` now defaults to True, so MRv2 is the default path. Set it False in `additional_config` to fall back to v1. Pooling models always use the v1 runner.

### What is included
* Phase 1 `TTModelState`, Phase 2 `TTRequestState` plus `TTInputBatch`, Phase 3 `TTModelRunnerV2` (host state machine: lifecycle, decode first batch select, host input and attn prep, two phase `execute_model`/`sample_tokens`, KV alloc, compiled forward plus sample, warmup).
* Multi device: SPMD device mesh, weight sharding, forward/KV/DP input sharding for TP, DP, and DP+TP.
* Features: prompt logprobs, cpu_sampling (Gumbel max host sampling), text only mrope, LoRA (mixin driven), multimodal `get_mm_embeddings`, structured output and grammar (bitmask masking), plus operator logging.
* Separately compiled forward->logits and sampling graphs, so post processing no longer re-specializes the model forward.

### 0.25.1 adaptations

* `TTModelState`: sets `inputs_embeds_size` itself (the 0.25.1 `ModelState` base sets it in its concrete `__init__`, which TT does not call because it would build a CUDA `EncoderRunner`), and `postprocess_state` matches the new `(idx_mapping, num_sampled, num_computed_tokens)` signature.
* `SamplingBatchView`: the uplift extended `XLASupportedSamplingMetadata.from_input_batch` to read `logits_processing_needs_token_ids`, `spec_token_ids`, `logitsprocs`, and `logitsprocs_need_output_token_ids`. The v2 batch view now exposes all four with the values the v2 generation path implies: pooling-only token-id needs are always False, `num_speculative_steps` is pinned to 0 so there are no drafts, and neither TT runner builds custom logits processors.
* `allowed_token_ids`: the uplift split the mask into a bool `allowed_token_ids_mask` (upstream contract) and a float `allowed_token_ids_additive_mask`. `sampler.py` already prefers the additive variant, so v2 keeps the additive path that avoids on-device `masked_fill` with bool tensors; only the CPU test's assertions needed widening to cover both.
* Speculative decode now raises instead of silently degrading (see below).

### Validation

* 109 MRv2 CPU tests pass on 0.25.1 (`tests/integrations/vllm_plugin/mrv2/`).
* Single device prefill + decode smoke runs on TT hardware on 0.25.1.
* The generative/sampling parity sweeps below were run on the 0.22.1 base; re-validation on 0.25.1 is via this PR's CI run.
  * Generative and sampling parity sweeps green on a single device n150 CI runner under MRv2 (v2 matches v1).
  * grammar and structured output: TP validated (a phone number regex produces the expected formatted output).
  * prompt logprobs and cpu_sampling: TP validated (v2 matches v1).
  * mrope: passes on a single device n150 CI runner.
  * Gemma 4 text only: passes on a single device Blackhole under MRv2.

### Known gaps (present on v1 too, not blockers)
* Gemma 4 multimodal image path OOMs on a single device: the vision forward needs more per bank DRAM than remains after the unsharded weights fill the device. Reproduces on v1 identically, so v2 does not regress it. Tracked separately; the image path already runs at TP=8 (#5257).
* LoRA on device e2e deferred: LoRA lowering is broken plugin wide on 0.20.2 and 0.22.1 and fails on v1 too, so v2 as default does not regress it.
* Speculative decode: the v2 runner pins `num_speculative_steps=0`, so a `speculative_config` would have had its drafts silently dropped now that v2 is the default. It raises `NotImplementedError` pointing at `use_v2_model_runner=False`. The NGram spec decode work in #5542 is v1 only and has no e2e test, so nothing regresses; porting it to v2 is tracked in #5795 Phase 3.
* Encoder decoder attention and multi type hybrid KV cache raise `NotImplementedError` guards, matching v1.
