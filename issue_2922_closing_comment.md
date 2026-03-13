## Summary

All user-facing [`SamplingParams`](https://docs.vllm.ai/en/v0.16.0/api/vllm/sampling_params/) fields have been swept against vLLM v0.16.0 (our current version, a superset of the v0.6.0 referenced in the issue description). #3255 (merged Feb 13) removed the greedy-only sampling hack and enabled the initial sweep test and #3652 (Mar 12) added prompt_logprobs, the last param requiring implementation for the current vLLM v0.16.0 version. Every param that executes on device has both E2E and synthetic test coverage, except `structured_outputs` which has E2E only (bitmask unpacking on device, but no standalone synthetic test). CPU-only params have E2E coverage. The one remaining gap (`repetition_detection`) is new in v0.17.0 and tracked in #3708. The v0.17.0 uplift is in progress via #3660, which also includes the `repetition_detection` smoke test.

### Testing strategy

- **E2E tests** (`test_sampling_params.py`) — spin up a real vLLM engine on TT hardware, issue requests with specific `SamplingParams`, and validate outputs. Covers both single-device and TP (n300-llmbox) fixtures.
- **Synthetic on-device tests** (`test_sampling_params_synthetic.py`) — compile and execute individual sampling graph stages (penalties, masking, top-k/top-p, Gumbel-max, gather) at production tensor shapes on device, without a full model. Catches device-level correctness issues that E2E tests may miss.
- **CPU correctness tests** (`test_logprobs_correctness.py`) — pure-CPU validation of logprobs math (log-softmax properties, gather ordering, rank correctness).
- **Output coherence test** (`test_output_coherence`) — greedy regression test against 6 simple prompts to catch gibberish output, runs on `push`.
- **Production vocab sizes on device** — synthetic tests are parametrized across production vocab sizes (128k Llama3, 152k Qwen3, 201k GPT-OSS) and run the full sampling pipeline (penalties, masking, top-k/top-p, softmax, argmax/Gumbel-max) as compiled `torch.compile` graphs on device.

> **Note:** Some of this work required tt-mlir compiler fixes (e.g. sort, softmax). Those fixes are not listed here but are tracked within the relevant tt-xla PRs and sub-issues.

## Coverage

| Name | Status | Test Coverage | On Device | PR | Notes |
|---|---|---|---|---|---|
| `temperature` | Done | E2E + synthetic | Yes | #3255, #3367 | |
| `top_k` | Done | E2E + synthetic | Yes | #3255, #3367 | |
| `top_p` | Done | E2E + synthetic | Yes | #3255, #3367 | |
| `min_p` | Done | E2E + synthetic | Yes | #3255, #3367 | |
| `presence_penalty` | Done | E2E + synthetic | Yes | #3255, #3370 | |
| `frequency_penalty` | Done | E2E + synthetic | Yes | #3255, #3370 | |
| `repetition_penalty` | Done | E2E + synthetic | Yes | #3255, #3370 | |
| `seed` | Done | E2E + synthetic | Yes | #3367, #3487 | Noise CPU, Gumbel-max on device |
| `logprobs` | Done | E2E + synthetic | Yes | #3255, #3416, #3652 | |
| `prompt_logprobs` | Done | E2E + synthetic | Yes | #3652 | log_softmax + gather on device |
| `logit_bias` | Done | E2E + synthetic | Yes | #3367, #3415 | |
| `bad_words` | Done | E2E + synthetic | Yes | #3367, #3415, #3482 | Multi-token support in #3482 |
| `allowed_token_ids` | Done | E2E + synthetic | Yes | #3564 | |
| `min_tokens` | Done | E2E + synthetic | Yes | #3564 | |
| `structured_outputs` | Done | E2E | Yes | #3581 | Bitmask unpacking on device |
| `stop` | Done | E2E | CPU-only (expected) | #3255 | String matching in decode loop |
| `stop_token_ids` | Done | E2E | CPU-only (expected) | #3367 | Token ID check in decode loop |
| `ignore_eos` | Done | E2E | CPU-only (expected) | #3367 | Flag controlling decode loop |
| `max_tokens` | Done | E2E | CPU-only (expected) | #3255 | Loop counter |
| `n` | Done | E2E | CPU-only (expected) | #3255 | Engine-level multiplexing |
| `include_stop_str_in_output` | Done | E2E | CPU-only (expected) | #3581 | Upstream vLLM |
| `detokenize` | Done | E2E | CPU-only (expected) | #3581 | Upstream vLLM |
| `skip_special_tokens` | Done | E2E | CPU-only (expected) | #3581 | Upstream vLLM |
| `spaces_between_special_tokens` | Done | E2E | CPU-only (expected) | #3581 | Upstream vLLM |
| `truncate_prompt_tokens` | Done | E2E | CPU-only (expected) | #3581 | Upstream vLLM. Removed in v0.17.0. |
| `output_kind` | Done | E2E | CPU-only (expected) | #3581 | Upstream vLLM |
| `repetition_detection` | Tracked | N/A | CPU-only (expected) | #3660 | New in v0.17.0. Smoke test ready, merges with uplift. #3708 |
| `logits_processors` | Won't implement | N/A | N/A | N/A | Removed in vLLM v0.17.0. Built-in processors already covered. |
| `best_of` | N/A | N/A | N/A | N/A | Removed in vLLM v1 |
| `skip_clone` | No test needed | N/A | N/A | N/A | Internal deep-copy optimization. No observable output change, no TT code path. |
| `flat_logprobs` | No test needed | N/A | N/A | N/A | Changes logprobs container type. No observable output change, no TT code path. |
| `extra_args` | No test needed | N/A | N/A | N/A | Pass-through dict for backend extensions. Unused by our plugin. |
| `skip_reading_prefix_cache` | No test needed | N/A | N/A | N/A | Internal prefix cache hint. No observable output change, no TT code path. |
| `output_text_buffer_length` | No test needed | N/A | N/A | N/A | Computed from stop strings. Upstream vLLM detokenizer, no TT code path. |

### Related fixes

| Item | PR |
|---|---|
| Remove greedy-only hack, initial sweep test | #3029, #3255 |
| Re-enable n300-llmbox TP tests | #3558 |
| Fix `sample_from_logits` + `gather_logprobs` compilation under SPMD | #3675 |
| Output coherence regression test (`test_output_coherence`) | #3675 |
| TP fixture switched to `ParallelLMHead` model (TinyLlama-1.1B) | #3653 |
