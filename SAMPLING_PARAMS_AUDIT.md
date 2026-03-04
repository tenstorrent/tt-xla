# SamplingParams Audit — TT-XLA vLLM Plugin (Mar 3 2026)

## Phase 1

| Param | Is SamplingParam? | E2E test (`test_sampling_params.py`) | Synthetic test (`test_sampling_params_synthetic.py`) | Executes on TT device? | Further work needed | Notes |
|---|---|---|---|---|---|---|
| **n** | Yes | `test_sampling_has_diversity_when_temp_positive` uses `n=4` | No (single sample per call) | N/A — vLLM engine-level param, schedules multiple samples per request. Sampling itself runs on device. | None | Engine-level multiplexing; each sample uses the on-device sampler |
| **presence_penalty** | Yes | `test_sampling_param_sweep`, `test_additive_penalties_end_to_end` | `test_penalties` | Yes — `apply_penalties()` in `sampler.py:84-124`, compiled graph | None | Mask tensors (`output_token_counts`) built on CPU, transferred to device. Math runs on device. |
| **frequency_penalty** | Yes | `test_sampling_param_sweep`, `test_additive_penalties_end_to_end` | `test_penalties` | Yes — same `apply_penalties()` path | None | Same pattern: count tensor built CPU-side, penalty math on device |
| **repetition_penalty** | Yes | `test_sampling_param_sweep`, `test_repetition_penalty_end_to_end` | `test_penalties` | Yes — same `apply_penalties()` path | None | Covers prompt ∪ output tokens via `prompt_token_mask` |
| **seed** | Yes | `test_seed` (same seed = same output, different seed = different output) | `test_seed_precomputed_noise`, `test_seed_mixed_batch` | Partially — `q_samples` (exponential noise) is pre-computed on **CPU** via `torch.Generator`, then transferred to device. The `probs.div_(q).argmax()` Gumbel-max trick runs **on device**. | Could move noise generation to device RNG, but CPU pre-computation is intentional (per-request Generator objects are CPU-only). No correctness gap. | CPU generators required for per-request seed isolation |
| **logprobs** | Yes | `test_logprobs` (structure, values, ranks) | `test_gather_logprobs`, `test_gather_logprobs_rank_nonzero_outside_topk`, `test_gather_logprobs_topk_indices_exact_on_device` | Yes — `compute_logprobs()` (log_softmax) and `gather_logprobs()` (topk + gather) run on device | `gather_logprobs` is currently **not** wrapped in `torch.compile` (commented out at `model_runner.py:2027`) — runs eagerly on XLA device. Compiling it would improve perf. Also `count_tokens_ge` has workaround for #3464 (bf16 bool fusion bug). | See `sampler.py:14-33` for the workaround |
| **temperature** | Yes | `test_sampling_param_sweep`, `test_greedy_determinism`, `test_combined_sampling` | `test_greedy`, `test_non_greedy`, `test_combined`, `test_boundary_values` | Yes — `apply_temperature()` in `sampler.py:70-79`, on-device div | None | |
| **top_p** | Yes | `test_sampling_param_sweep`, `test_combined_sampling` | `test_non_greedy`, `test_combined`, `test_boundary_values` | Yes — `apply_top_k_top_p()` in `sampler.py:284-328`, uses sort + cumsum + masking on device | `sort` has known issues at large vocab sizes (#3464 area). Uses TPU-optimized path (no scatter). | |
| **top_k** | Yes | `test_sampling_param_sweep`, `test_combined_sampling` | `test_non_greedy`, `test_combined`, `test_boundary_values` | Yes — same `apply_top_k_top_p()` function, sort + gather + masking on device | Same `sort` caveats as top_p | |
| **messages** | **No** — chat completions API param, not a SamplingParam | N/A | N/A | N/A — handled by the vLLM chat template / tokenizer layer on CPU before any model execution | None | Tokenized to `input_ids` before reaching the model runner |
| **stop** | Yes (string stop sequences) | `test_stop_sequences` | No — requires full decode loop | **CPU** — vLLM engine checks stop strings against decoded text after each generated token on CPU | None — stop string matching is inherently a CPU post-processing operation (string comparison against tokenizer output) | |
| **max_tokens** | Yes | `test_output_length_controls`, used in nearly every test | N/A (controls decode loop length) | **CPU** — vLLM engine counts generated tokens and stops the loop | None — correctly a CPU-side loop counter | |

## Phase 2

| Param | Is SamplingParam? | E2E test | Synthetic test | Executes on TT device? | Further work needed | Notes |
|---|---|---|---|---|---|---|
| **best_of** | **No** — removed in vLLM v1 | N/A | N/A | N/A | None | v0-only feature, fully removed. `SamplingParams(best_of=...)` raises `TypeError` in vLLM 0.15.0. Only a stale docstring in `llm.py` references it. |
| **min_p** | Yes | `test_sampling_param_sweep`, `test_combined_sampling` | `test_combined`, `test_boundary_values` | Yes — `apply_min_p()` in `sampler.py:243-261`, softmax + adaptive threshold masking on device | None | |
| **bad_words** | Yes | `test_bad_words` | `test_bad_words` | Partially — mask tensor (`bad_words_mask`) built on **CPU** (`_compute_bad_words_mask` in `metadata.py:125-166`), then transferred to device. The `-inf` masking (`logits + mask`) runs **on device**. Multi-token prefix matching runs on CPU per step. | CPU prefix matching is unavoidable (requires output history comparison). Single-token bans are fully device-side once the mask is built. | |
| **structured_outputs** (guided decoding) | Yes (via `guided_decoding` param) | `test_structured_outputs_regex` | **No** | **CPU** — `apply_grammar_bitmask()` at `model_runner.py:2063-2082` explicitly does `.to("cpu")`, unpacks the bitmask on CPU, then sends back to device. The `structured_decode` `torch.where` wrapper is compiled but the actual bitmask logic is CPU. | Moving bitwise ops to device would improve latency. | Supported but slow path; `apply_grammar_bitmask` is a known CPU bottleneck |
| **logit_bias** | Yes | `test_logit_bias` | `test_logit_bias` | Yes — `apply_logit_bias()` in `sampler.py:133-138`, simple `logits + bias_tensor` on device. Bias tensor built on CPU and transferred. | None | |
| **stop_token_ids** | Yes | `test_stop_token_ids` | No — requires decode loop | **CPU** — vLLM engine checks sampled token ID against stop set after each step on CPU | None — same as `stop`, correctly CPU post-processing | Tracked alongside `min_tokens` in `input_batch.py:162-163` |
| **ignore_eos** | Yes | `test_ignore_eos` | No — requires decode loop | **CPU** — vLLM engine-level: simply doesn't check for EOS token when `ignore_eos=True` | None — CPU-side flag controlling the decode loop | |

## Test Coverage Summary

| Param | E2E test (`test_sampling_params.py`) | Synthetic test (`test_sampling_params_synthetic.py`) | Notes |
|---|---|---|---|
| **temperature** | `test_sampling_param_sweep`, `test_greedy_determinism`, `test_combined_sampling` | `test_greedy`, `test_non_greedy`, `test_combined`, `test_boundary_values` | |
| **top_k** | `test_sampling_param_sweep`, `test_combined_sampling` | `test_non_greedy`, `test_combined` | |
| **top_p** | `test_sampling_param_sweep`, `test_combined_sampling` | `test_non_greedy`, `test_combined` | |
| **min_p** | `test_sampling_param_sweep`, `test_combined_sampling` | `test_combined`, `test_boundary_values` | |
| **presence_penalty** | `test_additive_penalties_end_to_end` | `test_penalties` | |
| **frequency_penalty** | `test_additive_penalties_end_to_end` | `test_penalties` | |
| **repetition_penalty** | `test_repetition_penalty_end_to_end` | `test_penalties` | |
| **seed** | `test_seed` | `test_seed_precomputed_noise`, `test_seed_mixed_batch` | |
| **logprobs** | `test_logprobs` | `test_gather_logprobs`, `test_gather_logprobs_rank_nonzero_outside_topk`, `test_gather_logprobs_topk_indices_exact_on_device` | |
| **bad_words** | `test_bad_words` | `test_bad_words` | |
| **logit_bias** | `test_logit_bias` | `test_logit_bias` | |
| **structured_outputs** | `test_structured_outputs_regex` | N/A | No device graph to test synthetically — bitmask unpacking runs on CPU |
| **stop** | `test_stop_sequences` | N/A | CPU-only (string matching in decode loop) |
| **stop_token_ids** | `test_stop_token_ids` | N/A | CPU-only (token ID check in decode loop) |
| **ignore_eos** | `test_ignore_eos` | N/A | CPU-only (flag controlling decode loop) |
| **max_tokens** | `test_output_length_controls` | N/A | CPU-only (loop counter) |
| **n** | `test_sampling_has_diversity_when_temp_positive` | N/A | Engine-level multiplexing |
| **min_tokens** | N/A | N/A | CPU-only (engine won't stop until min reached) |
| **best_of** | N/A | N/A | Removed in vLLM v1 |
| **messages** | N/A | N/A | Chat API param, not a sampling param |

### Not tested — not functional or not applicable (vLLM 0.15.0)

| Param | Status | Notes |
|---|---|---|
| **prompt_logprobs** | Plumbed but stubbed | `num_prompt_logprobs` tracked in `input_batch.py`/`model_runner.py`, but output always returns `None` (line 1340). Not implemented in TT plugin. |
| **allowed_token_ids** | Plumbed but not wired | Mask built in `input_batch.py` (lines 310-330), passed through metadata, but sampler never reads it. `metadata.py:62` sets `allowed_token_ids_mask = None`. |
| **logits_processors** | CPU callbacks | Custom callables, no device graph involvement. |
| **detokenize** | CPU post-processing | Output formatting only. |
| **skip_special_tokens** | CPU post-processing | Output formatting only. |
| **spaces_between_special_tokens** | CPU post-processing | Output formatting only. |
| **include_stop_str_in_output** | CPU post-processing | Output formatting only. |
| **truncate_prompt_tokens** | CPU pre-processing | Input truncation before model execution. |
| **flat_logprobs** | Output formatting | Controls logprobs output shape. |
| **output_kind** | Output formatting | Controls `RequestOutput` structure. |
| **skip_clone** / **skip_reading_prefix_cache** / **extra_args** | Internal plumbing | No user-facing behavior to test. |

Every param that executes on device has both e2e and synthetic coverage. CPU-only params have e2e coverage only, which is appropriate since there is no device graph to test.

## Execution Summary

**Fully on device** (compiled graph): `temperature`, `top_k`, `top_p`, `min_p`, `presence_penalty`, `frequency_penalty`, `repetition_penalty`, `logit_bias`, `bad_words` (mask application), greedy argmax

**CPU pre-computation, device execution**: `seed` (noise generation CPU, Gumbel-max on device), `bad_words` (mask construction CPU, application on device), penalties (count/mask tensors CPU, penalty math on device)

**Correctly CPU-only** (engine-level): `stop`, `stop_token_ids`, `ignore_eos`, `max_tokens`, `n`, `messages`

**Not applicable**: `best_of` (removed in vLLM v1)

**CPU bottleneck / needs work**:
1. **`structured_outputs`** — grammar bitmask unpacking runs entirely on CPU (`apply_grammar_bitmask` line 2063-2082). E2e test added (`a87ab8fca`).
2. ~~**`logprobs` gather** — `gather_logprobs` is not `torch.compile`d (commented out at `model_runner.py:2027`), runs eagerly.~~ **DONE** — enabled `torch.compile` in `9fb0445b2`.
3. ~~**`sample_from_logits`** — also not `torch.compile`d (commented out at `model_runner.py:2005-2007`) due to SPMD correctness issue. Runs eagerly on XLA device.~~ **DONE** — enabled `torch.compile` for SPMD mode in `9fb0445b2`.

## Action Items — Completed

1. ~~**Add `structured_outputs` e2e test**~~ — **DONE** (`a87ab8fca`)
3. ~~**Re-enable `torch.compile` for `sample_from_logits`**~~ — **DONE** (`9fb0445b2`)
4. ~~**Re-enable `torch.compile` for `gather_logprobs`**~~ — **DONE** (`9fb0445b2`)

## Action Items — Remaining

### Broken / stubbed params (plumbed but not functional)

5. **Implement `allowed_token_ids`** — Mask is built in `input_batch.py:310-330` and transferred to device, but the sampler never applies it. Needs an `apply_allowed_tokens()` call in `sampler.py` (similar to `apply_logit_bias`). Add e2e + synthetic tests.
6. **Implement `prompt_logprobs`** — `num_prompt_logprobs` is tracked in `input_batch.py`/`model_runner.py`, but `model_runner.py:1340` hardcodes the result to `None`. Needs actual log_softmax computation on prompt tokens and gathering of top-k. Add e2e test.
7. **Implement `min_tokens`** — Stored in `input_batch.py:162-163` but `metadata.py:54` marks it `None` with comment "impl is not vectorized". Never enforced — engine can stop before min_tokens is reached. Add e2e test.
8. **Implement `logits_processors`** — Stored in `input_batch.py:199-200` and passed to `SamplingMetadata`, but never called in `sampler.py`. Needs a hook in the sampling pipeline to apply custom processor callables. Add e2e test.

### Performance improvements

2. **Move `apply_grammar_bitmask` to device** — currently explicitly CPU-bound (`model_runner.py:2062-2076`), would benefit structured output latency.
9. **Fix #3464** (bf16 bool fusion) — would remove `count_tokens_ge` workaround in `sampler.py` and enable `clamp(min=1)` directly.

### Params handled by upstream vLLM (verify, don't implement)

These are CPU-side flags that should fall through to upstream vLLM without TT plugin involvement. Verify they work correctly and add e2e tests if missing.

10. **Verify `include_stop_str_in_output`** — CPU post-processing flag. Should work via upstream vLLM. Add e2e test to confirm.
11. **Verify `detokenize`** — CPU post-processing flag. Should work via upstream. Add e2e test to confirm.
12. **Verify `skip_special_tokens`** — CPU post-processing flag. Should work via upstream. Add e2e test to confirm.
13. **Verify `spaces_between_special_tokens`** — CPU post-processing flag. Should work via upstream. Add e2e test to confirm.
14. **Verify `truncate_prompt_tokens`** — CPU pre-processing flag. Should work via upstream scheduler. Add e2e test to confirm.
15. **Verify `output_kind`** — CPU output format control. Should work via upstream. Add e2e test to confirm.

### Not a SamplingParam

16. **`reasoning_effort`** — OpenAI chat completions API param (`ChatCompletionRequest.reasoning_effort`), not a `SamplingParams` field. Modifies the system prompt via chat template. Handled in `vllm/entrypoints/openai/`. Needs verification with TT serving endpoint if applicable.
17. **`guided_decoding`** — Old vLLM API name, replaced by `structured_outputs` (`StructuredOutputsParams`) in vLLM 0.15.0. Already tested via `test_structured_outputs_regex`.
18. **`extra_args`** — Pass-through dict for backend-specific extensions. No standard behavior to test.
