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
| **structured_outputs** (guided decoding) | Yes (via `guided_decoding` param) | **No dedicated test** | **No** | **CPU** — `apply_grammar_bitmask()` at `model_runner.py:2063-2082` explicitly does `.to("cpu")`, unpacks the bitmask on CPU, then sends back to device. The `structured_decode` `torch.where` wrapper is compiled but the actual bitmask logic is CPU. | Major gap: grammar bitmask unpacking runs entirely on CPU. Moving bitwise ops to device would improve latency. Needs dedicated test coverage. | Supported but slow path; `apply_grammar_bitmask` is a known CPU bottleneck |
| **logit_bias** | Yes | `test_logit_bias` | `test_logit_bias` | Yes — `apply_logit_bias()` in `sampler.py:133-138`, simple `logits + bias_tensor` on device. Bias tensor built on CPU and transferred. | None | |
| **stop_token_ids** | Yes | `test_stop_token_ids` | No — requires decode loop | **CPU** — vLLM engine checks sampled token ID against stop set after each step on CPU | None — same as `stop`, correctly CPU post-processing | Tracked alongside `min_tokens` in `input_batch.py:162-163` |
| **ignore_eos** | Yes | `test_ignore_eos` | No — requires decode loop | **CPU** — vLLM engine-level: simply doesn't check for EOS token when `ignore_eos=True` | None — CPU-side flag controlling the decode loop | |

## Execution Summary

**Fully on device** (compiled graph): `temperature`, `top_k`, `top_p`, `min_p`, `presence_penalty`, `frequency_penalty`, `repetition_penalty`, `logit_bias`, `bad_words` (mask application), greedy argmax

**CPU pre-computation, device execution**: `seed` (noise generation CPU, Gumbel-max on device), `bad_words` (mask construction CPU, application on device), penalties (count/mask tensors CPU, penalty math on device)

**Correctly CPU-only** (engine-level): `stop`, `stop_token_ids`, `ignore_eos`, `max_tokens`, `n`, `messages`

**Not applicable**: `best_of` (removed in vLLM v1)

**CPU bottleneck / needs work**:
1. **`structured_outputs`** — grammar bitmask unpacking runs entirely on CPU (`apply_grammar_bitmask` line 2063-2082). No test coverage.
2. **`logprobs` gather** — `gather_logprobs` is not `torch.compile`d (commented out at `model_runner.py:2027`), runs eagerly.
3. **`sample_from_logits`** — also not `torch.compile`d (commented out at `model_runner.py:2005-2007`) due to SPMD correctness issue. Runs eagerly on XLA device.

## Action Items

1. **Add `structured_outputs` e2e test** — supported but untested, runs on CPU
2. **Move `apply_grammar_bitmask` to device** — currently explicitly CPU-bound, would benefit structured output latency
3. **Re-enable `torch.compile` for `sample_from_logits`** — blocked on SPMD correctness fix (see TODO at `model_runner.py:2005`)
4. **Re-enable `torch.compile` for `gather_logprobs`** — commented out at `model_runner.py:2027`
5. **Fix #3464** (bf16 bool fusion) — would remove `count_tokens_ge` workaround and enable `clamp(min=1)` directly
