# SamplingParams Audit — TT-XLA vLLM Plugin (updated Mar 9 2026)

## Test Coverage Summary

| Param | E2E test (`test_sampling_params.py`) | Synthetic test (`test_sampling_params_synthetic.py`) | Notes |
|---|---|---|---|
| **temperature** | `test_sampling_param_sweep`, `test_greedy_determinism`, `test_combined_sampling` | `test_greedy`, `test_non_greedy`, `test_combined`, `test_boundary_values` | |
| **top_k** | `test_sampling_param_sweep`, `test_combined_sampling` | `test_non_greedy`, `test_combined` | |
| **top_p** | `test_sampling_param_sweep`, `test_combined_sampling` | `test_non_greedy`, `test_combined` | |
| **min_p** | `test_sampling_param_sweep`, `test_combined_sampling` | `test_combined`, `test_boundary_values` | |
| **presence_penalty** | `test_sampling_param_sweep`, `test_additive_penalties_end_to_end` | `test_penalties` | |
| **frequency_penalty** | `test_sampling_param_sweep`, `test_additive_penalties_end_to_end` | `test_penalties` | |
| **repetition_penalty** | `test_sampling_param_sweep`, `test_repetition_penalty_end_to_end` | `test_penalties` | |
| **seed** | `test_seed` | `test_seed_precomputed_noise`, `test_seed_mixed_batch` | |
| **logprobs** | `test_logprobs` | `test_gather_logprobs`, `test_gather_logprobs_rank_nonzero_outside_topk`, `test_gather_logprobs_topk_indices_exact_on_device` | |
| **bad_words** | `test_bad_words` | `test_bad_words` | |
| **logit_bias** | `test_logit_bias` | `test_logit_bias` | |
| **allowed_token_ids** | `test_allowed_token_ids` | `test_allowed_token_ids`, `test_allowed_token_ids_mixed_batch` | Implemented in PR #3564 |
| **min_tokens** | `test_min_tokens` | `test_min_tokens` | Implemented in PR #3564 |
| **structured_outputs** | `test_structured_outputs_regex` | N/A | Bitmask unpacking moved to device in PR #3581 |
| **stop** | `test_stop_sequences` | N/A | CPU-only (string matching in decode loop) |
| **stop_token_ids** | `test_stop_token_ids` | N/A | CPU-only (token ID check in decode loop) |
| **ignore_eos** | `test_ignore_eos` | N/A | CPU-only (flag controlling decode loop) |
| **max_tokens** | `test_output_length_controls` | N/A | CPU-only (loop counter) |
| **n** | `test_sampling_has_diversity_when_temp_positive` | N/A | Engine-level multiplexing |
| **include_stop_str_in_output** | `test_include_stop_str_in_output` | N/A | CPU-only (upstream vLLM) |
| **detokenize** | `test_detokenize` | N/A | CPU-only (upstream vLLM) |
| **skip_special_tokens** | `test_skip_special_tokens` | N/A | CPU-only (upstream vLLM) |
| **spaces_between_special_tokens** | `test_spaces_between_special_tokens` | N/A | CPU-only (upstream vLLM) |
| **truncate_prompt_tokens** | `test_truncate_prompt_tokens` | N/A | CPU-only (upstream vLLM) |
| **output_kind** | `test_output_kind` | N/A | CPU-only (upstream vLLM) |
| **best_of** | N/A | N/A | Removed in vLLM v1 |
| **messages** | N/A | N/A | Chat API param, not a sampling param |

### Not tested — not functional or not applicable

| Param | Status | Notes |
|---|---|---|
| **prompt_logprobs** | Plumbed but stubbed | `num_prompt_logprobs` tracked in `input_batch.py`/`model_runner.py`, but output always returns `None`. Not implemented in TT plugin. |
| **logits_processors** | CPU callbacks | Custom callables, stored but never applied in sampler. |
| **flat_logprobs** | Output formatting | Controls logprobs output shape. |
| **skip_clone** / **skip_reading_prefix_cache** / **extra_args** | Internal plumbing | No user-facing behavior to test. |

Every param that executes on device has both E2E and synthetic coverage. CPU-only params have E2E coverage only, which is appropriate since there is no device graph to test.

## Execution Summary

**Fully on device** (compiled graph): `temperature`, `top_k`, `top_p`, `min_p`, `presence_penalty`, `frequency_penalty`, `repetition_penalty`, `logit_bias`, `bad_words` (mask application), `allowed_token_ids` (mask application), `structured_outputs` (bitmask unpacking + masking), greedy argmax

**CPU pre-computation, device execution**: `seed` (noise generation CPU, Gumbel-max on device), `bad_words` (mask construction CPU, application on device), `allowed_token_ids` (mask construction CPU, application on device), `min_tokens` (stop token mask built CPU, applied on device), penalties (count/mask tensors CPU, penalty math on device)

**Correctly CPU-only** (engine-level): `stop`, `stop_token_ids`, `ignore_eos`, `max_tokens`, `n`, `messages`, `include_stop_str_in_output`, `detokenize`, `skip_special_tokens`, `spaces_between_special_tokens`, `truncate_prompt_tokens`, `output_kind`

**Not applicable**: `best_of` (removed in vLLM v1)

## Action Items — Completed

1. ~~**Add `structured_outputs` E2E test**~~ — **DONE** (PR #3581)
2. ~~**Move `apply_grammar_bitmask` to device**~~ — **DONE** (PR #3581)
3. ~~**Re-enable `torch.compile` for `gather_logprobs`**~~ — **DONE** (PR #3581)
4. ~~**Implement `allowed_token_ids`**~~ — **DONE** (PR #3564)
5. ~~**Implement `min_tokens`**~~ — **DONE** (PR #3564)
6. ~~**Verify upstream vLLM CPU-side flags**~~ (`include_stop_str_in_output`, `detokenize`, `skip_special_tokens`, `spaces_between_special_tokens`, `truncate_prompt_tokens`, `output_kind`) — **DONE** (PR #3581)
7. ~~**Add degenerate output guard to `test_greedy_determinism`**~~ — **DONE** (PR #3581)

## Action Items — Remaining

### Broken / stubbed params

8. **Implement `prompt_logprobs`** — `num_prompt_logprobs` is tracked but output is hardcoded to `None`. Needs log_softmax computation on prompt tokens and top-k gathering.
9. **Implement `logits_processors`** — Stored and passed to `SamplingMetadata` but never called in `sampler.py`. Needs a hook to apply custom processor callables.

### Performance / correctness

10. **Fix `sample_from_logits` compilation under SPMD** (#3589) — `sample_from_logits` is only compiled when `enable_tensor_parallel` is False. Under SPMD/TP mode it runs eagerly due to a correctness issue (all tokens become token_id 0 when compiled).
11. **Fix #3464** (bf16 bool fusion) — would remove `count_tokens_ge` workaround in `sampler.py`.
12. **Add TP test fixture with `ParallelLMHead`** (#3590) — both n300 fixtures use tied-embedding models, so the `sharding_constraint_tensor` logit replication path is never tested.

### Not a SamplingParam

13. **`reasoning_effort`** — OpenAI chat completions API param, not a `SamplingParams` field. Needs verification with TT serving endpoint if applicable.
14. **`guided_decoding`** — Old vLLM API name, replaced by `structured_outputs` in vLLM 0.15.0. Already tested.
15. **`extra_args`** — Pass-through dict for backend-specific extensions. No standard behavior to test.
