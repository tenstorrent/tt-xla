## Summary

All user-facing `SamplingParams` fields (vLLM v0.16.0) have been swept. Every param that executes on device has both E2E and synthetic test coverage. CPU-only params have E2E coverage. The one remaining gap (`repetition_detection`) is new in v0.17.0 and tracked in #3708.

### Testing strategy

- **E2E tests** (`test_sampling_params.py`) — spin up a real vLLM engine on TT hardware, issue requests with specific `SamplingParams`, and validate outputs. Covers both single-device and TP (n300-llmbox) fixtures.
- **Synthetic on-device tests** (`test_sampling_params_synthetic.py`) — compile and execute individual sampling graph stages (penalties, masking, top-k/top-p, Gumbel-max, gather) at production tensor shapes on device, without a full model. Catches device-level correctness issues that E2E tests may miss.
- **CPU correctness tests** (`test_logprobs_correctness.py`) — pure-CPU validation of logprobs math (log-softmax properties, gather ordering, rank correctness).
- **Output coherence test** (`test_output_coherence`) — greedy regression test against 6 simple prompts to catch gibberish output, runs on `push`.

> **Note:** Some of this work required tt-mlir compiler fixes (e.g. sort, softmax). Those fixes are not listed here but are tracked within the relevant tt-xla PRs and sub-issues.

## Coverage

| Name | Status | Test Coverage | On Device | PR | Notes |
|---|---|---|---|---|---|
| `temperature` | Done | E2E + synthetic | Yes | #3255, #3367 | |
| `top_k` | Done | E2E + synthetic | Yes | #3255, #3367 | |
| `top_p` | Done | E2E + synthetic | Yes | #3255, #3367 | |
| `min_p` | Done | E2E + synthetic | Yes | #3367 | |
| `presence_penalty` | Done | E2E + synthetic | Yes | #3370, #3367 | |
| `frequency_penalty` | Done | E2E + synthetic | Yes | #3370, #3367 | |
| `repetition_penalty` | Done | E2E + synthetic | Yes | #3370, #3367 | |
| `seed` | Done | E2E + synthetic | Yes | #3487, #3367 | Noise CPU, Gumbel-max on device |
| `logprobs` | Done | E2E + synthetic | Yes | #3416, #3652 | |
| `prompt_logprobs` | Done | E2E + synthetic | Yes | #3652 | log_softmax + gather on device |
| `logit_bias` | Done | E2E + synthetic | Yes | #3415, #3367 | |
| `bad_words` | Done | E2E + synthetic | Yes | #3415, #3482 | Multi-token support |
| `allowed_token_ids` | Done | E2E + synthetic | Yes | #3564 | |
| `min_tokens` | Done | E2E + synthetic | Yes | #3564 | |
| `structured_outputs` | Done | E2E | Yes | #3581 | Bitmask unpacking on device |
| `stop` | Done | E2E | CPU-only (expected) | #3581 | String matching in decode loop |
| `stop_token_ids` | Done | E2E | CPU-only (expected) | #3581 | Token ID check in decode loop |
| `ignore_eos` | Done | E2E | CPU-only (expected) | #3581 | Flag controlling decode loop |
| `max_tokens` | Done | E2E | CPU-only (expected) | #3581 | Loop counter |
| `n` | Done | E2E | CPU-only (expected) | #3255 | Engine-level multiplexing |
| `include_stop_str_in_output` | Done | E2E | CPU-only (expected) | #3581 | Upstream vLLM |
| `detokenize` | Done | E2E | CPU-only (expected) | #3581 | Upstream vLLM |
| `skip_special_tokens` | Done | E2E | CPU-only (expected) | #3581 | Upstream vLLM |
| `spaces_between_special_tokens` | Done | E2E | CPU-only (expected) | #3581 | Upstream vLLM |
| `truncate_prompt_tokens` | Done | E2E | CPU-only (expected) | #3581 | Upstream vLLM |
| `output_kind` | Done | E2E | CPU-only (expected) | #3581 | Upstream vLLM |
| `repetition_detection` | Tracked | N/A | CPU-only (expected) | #3660 | New in v0.17.0. Smoke test ready, merges with uplift. #3708 |
| `logits_processors` | Won't implement | N/A | N/A | N/A | Removed in vLLM v0.17.0. Built-in processors already covered. |
| `best_of` | N/A | N/A | N/A | N/A | Removed in vLLM v1 |

### Related fixes

| Item | PR |
|---|---|
| Remove greedy-only hack, initial sweep test | #3029, #3255 |
| Fix `sample_from_logits` + `gather_logprobs` compilation under SPMD | #3675 |
| Output coherence regression test (`test_output_coherence`) | #3675 |
| TP fixture switched to `ParallelLMHead` model (TinyLlama-1.1B) | #3653 |
