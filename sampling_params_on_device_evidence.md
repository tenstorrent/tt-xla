# SamplingParams On-Device Execution — Evidence Summary

This document records the evidence that all on-device SamplingParams execute on TT
hardware, with sampling completed before any tensor is transferred back to CPU.

## Key claim

When a sampling graph runs on TT device, the tensor transferred back to CPU via PJRT
`copyToHost` is shape `(batch_size, 1)` — one token ID per request — not
`(batch_size, vocab_size)` logits. This proves sampling (argmax, top-k/top-p filtering,
Gumbel-max, penalty application, masking) happens on device, and only the final token ID
crosses the device-to-host boundary.

## Three independent sources of evidence

All three were instrumented and observed in the same runs:

1. **Python `logger.info` in `model_runner.py`** — logs `sample_from_logits output:
shape=...` immediately before the `.cpu()` call that triggers the transfer, showing the
XLA tensor shape while it is still on device.

2. **C++ `LOG_F(INFO, ...)` in `buffer_instance.cc`** — logs `PJRT copyToHost:
shape=... size=... bytes` inside `BufferInstance::copyToHost()`, at the exact point the
PJRT runtime transfers the tensor from device to host. This is the lowest-level
confirmation available without modifying the TT-MLIR runtime.

3. **Shape probe in `scripts/verify_sampling_on_device.py`** — records `.shape` on the
CPU tensor returned by `torch.compile(backend="tt")` for each SamplingParam scenario.
Since the shape is set by the PJRT buffer dimensions, it agrees with the C++ log by
construction.

## E2E test results (primary evidence)

Run: `VLLM_LOGGING_LEVEL=INFO pytest tests/integrations/vllm_plugin/sampling/test_sampling_params.py -m single_device`

Model: OPT-125M (`vocab_size=50272`). **32 passed, 0 failed.**

Every decode step in every test produced:
- Python: `sample_from_logits output: shape=torch.Size([1, 1]) (input logits shape=torch.Size([1, 50272]))`
- C++: `PJRT copyToHost: shape=[1, 1] size=8 bytes`

| Test | Result | Output shape | Input logits shape |
|---|---|---|---|
| `test_sampling_param_sweep[temperature-*]` | PASSED | `[1, 1]` | `[1, 50272]` |
| `test_sampling_param_sweep[top_k-*]` | PASSED | `[1, 1]` | `[1, 50272]` |
| `test_sampling_param_sweep[top_p-*]` | PASSED | `[1, 1]` | `[1, 50272]` |
| `test_sampling_param_sweep[min_p-*]` | PASSED | `[1, 1]` | `[1, 50272]` |
| `test_sampling_param_sweep[presence_penalty-*]` | PASSED | `[1, 1]` | `[1, 50272]` |
| `test_sampling_param_sweep[frequency_penalty-*]` | PASSED | `[1, 1]` | `[1, 50272]` |
| `test_sampling_param_sweep[repetition_penalty-*]` | PASSED | `[1, 1]` | `[1, 50272]` |
| `test_sampling_has_diversity_when_temp_positive` | PASSED | `[1, 1]` | `[1, 50272]` |
| `test_greedy_determinism` | PASSED | `[1, 1]` | `[1, 50272]` |
| `test_output_coherence` | PASSED | `[1, 1]` | `[1, 50272]` |
| `test_combined_sampling` | PASSED | `[1, 1]` | `[1, 50272]` |
| `test_stop_sequences` | PASSED | `[1, 1]` | `[1, 50272]` |
| `test_logprobs` | PASSED | `[1, 1]` | `[1, 50272]` |
| `test_prompt_logprobs` | PASSED | `[1, 1]` | `[1, 50272]` |
| `test_output_length_controls` | PASSED | `[1, 1]` | `[1, 50272]` |
| `test_seed` | PASSED | `[1, 1]` | `[1, 50272]` |
| `test_bad_words` | PASSED | `[1, 1]` | `[1, 50272]` |
| `test_logit_bias` | PASSED | `[1, 1]` | `[1, 50272]` |
| `test_stop_token_ids` | PASSED | `[1, 1]` | `[1, 50272]` |
| `test_ignore_eos` | PASSED | `[1, 1]` | `[1, 50272]` |
| `test_additive_penalties_end_to_end` | PASSED | `[1, 1]` | `[1, 50272]` |
| `test_repetition_penalty_end_to_end` | PASSED | `[1, 1]` | `[1, 50272]` |
| `test_parameter_boundary_values` | PASSED | `[1, 1]` | `[1, 50272]` |
| `test_allowed_token_ids` | PASSED | `[1, 1]` | `[1, 50272]` |
| `test_min_tokens` | PASSED | `[1, 1]` | `[1, 50272]` |
| `test_structured_outputs_regex` | PASSED | `[1, 1]` | `[1, 50272]` |
| `test_include_stop_str_in_output` | PASSED | `[1, 1]` | `[1, 50272]` |
| `test_detokenize` | PASSED | `[1, 1]` | `[1, 50272]` |
| `test_skip_special_tokens` | PASSED | `[1, 1]` | `[1, 50272]` |
| `test_spaces_between_special_tokens` | PASSED | `[1, 1]` | `[1, 50272]` |
| `test_truncate_prompt_tokens` | PASSED | `[1, 1]` | `[1, 50272]` |
| `test_output_kind` | PASSED | `[1, 1]` | `[1, 50272]` |

### Non-`[1, 1]` transfers observed — all expected

| Shape | Size | Test | Explanation |
|---|---|---|---|
| `[1, 21]` × 2 | 84 bytes | `test_logprobs` | `gather_logprobs` output: `(batch=1, 20 logprobs + 1 sampled token)` |
| `[1]` | 4 bytes | `test_logprobs` | `selected_token_ranks` — one rank scalar per batch item |
| `[32, 768]` × 2 | 49152 bytes | `test_prompt_logprobs` | Hidden states for prompt logprobs: `max_num_reqs=32`, `hidden_dim=768` (OPT-125M). Captured on CPU, then passed back to device for `compute_logits`/`gather_logprobs`. |

No `(batch, vocab_size)` logits tensor was ever transferred to the host.

## Standalone script (second datapoint)

`scripts/verify_sampling_on_device.py` exercises each on-device SamplingParam
independently via `torch.compile(backend="tt")` at Llama-3 vocab size (128256), without
a full model. It records the input and output tensor shapes and checks that no output
dimension equals `vocab_size`. Results at `vocab_size=128256, batch_size=1`:

| SamplingParam | Input Shape | Output Shape(s) | On Device? |
|---|---|---|---|
| temperature | (1, 128256) | (1, 1) | YES |
| top_k | (1, 128256) | (1, 1) | YES |
| top_p | (1, 128256) | (1, 1) | YES |
| min_p | (1, 128256) | (1, 1) | YES |
| presence/frequency/repetition_penalty | (1, 128256) | (1, 1) | YES |
| logit_bias | (1, 128256) | (1, 1) | YES |
| bad_words | (1, 128256) | (1, 1) | YES |
| allowed_token_ids | (1, 128256) | (1, 1) | YES |
| min_tokens | (1, 128256) | (1, 1) | YES |
| seed | (1, 128256) | (1, 1) | YES |
| greedy (argmax) | (1, 128256) | (1, 1) | YES |
| logprobs | (2, 128256) | (2, 6), (2, 6), (2) | YES |
| prompt_logprobs | (8, 128256) | (8, 6), (8, 6), (8) | YES |
| structured_outputs | (1, 128256) | (1, 1) | YES |

The C++ `PJRT copyToHost` log confirmed matching shapes for every scenario.
