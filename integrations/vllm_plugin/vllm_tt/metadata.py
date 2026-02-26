# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass, field

import numpy as np
import torch

from .input_batch import InputBatch

DEFAULT_SAMPLING_PARAMS = dict(
    temperature=-1.0,
    min_p=0.0,
    # strictly disabled for now
    top_k=0,
    top_p=1.0,
    frequency_penalties=0.0,
    presence_penalties=0.0,
    repetition_penalties=1.0,
)


@dataclass
class XLASupportedSamplingMetadata:
    # This class exposes a more xla-friendly interface than SamplingMetadata
    # on TPU, in particular all arguments should be traceable and no optionals
    # are allowed, to avoid graph recompilation on Nones.
    temperature: torch.Tensor = None

    min_p: torch.Tensor = None
    top_k: torch.Tensor = None
    top_p: torch.Tensor = None

    all_greedy: bool = True
    all_random: bool = False

    # Whether logprobs are to be gathered in this batch of request. To balance
    # out compile time and runtime, a fixed `max_number_logprobs` value is used
    # when gathering logprobs, regardless of the values specified in the batch.
    logprobs: bool = False

    # Penalty support. no_penalties=True skips the penalty path entirely.
    no_penalties: bool = True
    output_token_counts: torch.Tensor | None = None
    # Bool mask [batch, vocab]: True for tokens that appeared in the prompt.
    # Used for repetition penalty (prompt ∪ output), matching the vLLM GPU spec.
    prompt_token_mask: torch.Tensor | None = None
    presence_penalties: torch.Tensor | None = None
    frequency_penalties: torch.Tensor | None = None
    repetition_penalties: torch.Tensor | None = None
    prompt_token_ids = None
    output_token_ids: list[list[int]] = field(default_factory=lambda: list())

    min_tokens = None  # impl is not vectorized

    logit_bias: list[dict[int, float] | None] = field(default_factory=lambda: list())

    allowed_token_ids_mask = None

    # Bool mask [batch, vocab]: True for tokens that should be banned this step.
    # Pre-computed on CPU via prefix-matching against output_token_ids,
    # then transferred to device for a single masked_fill_ in the sampler.
    no_bad_words: bool = True
    bad_words_mask: torch.Tensor | None = None

    # Generator not supported by xla
    _generators: dict[int, torch.Generator] = field(default_factory=lambda: dict())

    @property
    def generators(self) -> dict[int, torch.Generator]:
        # Generator not supported by torch/xla. This field must be immutable.
        return self._generators

    @staticmethod
    def _compute_token_counts(
        req_output_token_ids: list[list[int] | None],
        padded_num_reqs: int,
        vocab_size: int,
    ) -> torch.Tensor:
        """Build a [padded_num_reqs, vocab_size] float32 count tensor on CPU.

        Each entry [i, v] is the number of times token v appeared in the
        output generated so far for request i. Padding rows are all zeros.
        """
        counts = torch.zeros(padded_num_reqs, vocab_size, dtype=torch.float32)
        for i, token_ids in enumerate(req_output_token_ids[:padded_num_reqs]):
            if token_ids:
                for token_id in token_ids:
                    counts[i, token_id] += 1
        return counts

    @staticmethod
    def _compute_prompt_mask(
        num_prompt_tokens: np.ndarray,
        token_ids_cpu: np.ndarray,
        padded_num_reqs: int,
        vocab_size: int,
    ) -> torch.Tensor:
        """Build a [padded_num_reqs, vocab_size] bool mask for prompt tokens on CPU.

        Each entry [i, v] is True if token v appeared in the prompt for request i.
        Padding rows are all False. Used alongside output_token_counts so the
        repetition penalty covers prompt ∪ output, matching the vLLM GPU spec.
        """
        mask = torch.zeros(padded_num_reqs, vocab_size, dtype=torch.bool)
        for i in range(padded_num_reqs):
            n = int(num_prompt_tokens[i])
            if n > 0:
                prompt_ids = torch.from_numpy(token_ids_cpu[i, :n].copy()).long()
                valid_ids = prompt_ids[prompt_ids < vocab_size]
                if valid_ids.numel() > 0:
                    mask[i].scatter_(0, valid_ids, True)
        return mask

    @staticmethod
    def _compute_bad_words_mask(
        bad_words_token_ids: dict[int, list[list[int]]],
        req_output_token_ids: list[list[int] | None],
        padded_num_reqs: int,
        vocab_size: int,
    ) -> torch.Tensor:
        """Build a [padded_num_reqs, vocab_size] bool mask for banned tokens.

        For each request that has bad_words, perform prefix matching against
        its output token history.  A multi-token bad word ``[t0, t1, ..., tN]``
        bans ``tN`` only when the most recent N output tokens equal
        ``[t0, ..., tN-1]``.  Single-token bad words are always banned.
        Padding rows are all False.

        This mirrors ``vllm.v1.sample.ops.bad_words._apply_bad_words_single_batch``
        but materialises the result as a fixed-shape mask that can be applied
        on-device with a single ``masked_fill_``.
        """
        mask = torch.zeros(padded_num_reqs, vocab_size, dtype=torch.bool)
        for req_idx, bad_words_list in bad_words_token_ids.items():
            if req_idx >= padded_num_reqs:
                continue
            past = (
                req_output_token_ids[req_idx]
                if req_idx < len(req_output_token_ids)
                else None
            )
            past = past or []
            for bad_word_ids in bad_words_list:
                if len(bad_word_ids) == 0:
                    continue
                if len(bad_word_ids) > len(past) + 1:
                    continue  # not enough history to match prefix yet
                prefix_length = len(bad_word_ids) - 1
                last_token_id = bad_word_ids[-1]
                if last_token_id >= vocab_size:
                    continue
                if prefix_length == 0:
                    # Single-token bad word: always banned.
                    mask[req_idx, last_token_id] = True
                else:
                    actual_prefix = past[-prefix_length:]
                    expected_prefix = bad_word_ids[:prefix_length]
                    if actual_prefix == expected_prefix:
                        mask[req_idx, last_token_id] = True
        return mask

    @classmethod
    def from_input_batch(
        cls,
        input_batch: InputBatch,
        padded_num_reqs: int,
        xla_device: torch.device,
        generate_params_if_all_greedy: bool = False,
        vocab_size: int | None = None,
    ) -> "XLASupportedSamplingMetadata":
        """
        Copy sampling tensors slices from `input_batch` to on device tensors.

        `InputBatch._make_sampling_metadata` causes recompilation on XLA as it
        slices dynamic shapes on device tensors. This impl moves the dynamic
        ops to CPU and produces tensors of fixed `padded_num_reqs` size.

        Args:
            input_batch: The input batch containing sampling parameters.
            padded_num_reqs: The padded number of requests.
            xla_device: The XLA device.
            generate_params_if_all_greedy: If True, generate sampling parameters
                even if all requests are greedy. this is useful for cases where
                we want to pre-compile a graph with sampling parameters, even if
                they are not strictly needed for greedy decoding.
        """
        needs_logprobs = (
            input_batch.max_num_logprobs > 0 if input_batch.max_num_logprobs else False
        )

        # Guards for parameters that are tracked by InputBatch but not yet
        # forwarded through this function into the compiled sampler graph.
        # Each raise should be removed once the feature is fully plumbed here.
        if input_batch.generators:
            raise NotImplementedError(
                "seed is not yet supported in the TT sampler. "
                "Per-request generators are not available on TT devices. "
                "https://github.com/tenstorrent/tt-xla/issues/3365"
            )
        if any(
            input_batch.logit_bias[i] is not None for i in range(input_batch.num_reqs)
        ):
            raise NotImplementedError(
                "logit_bias is not yet supported in the TT sampler. "
                "https://github.com/tenstorrent/tt-xla/issues/3364"
            )
        has_bad_words = bool(input_batch.bad_words_token_ids)

        # Early return to avoid unnecessary cpu to tpu copy.
        # Must NOT skip when penalties or bad_words are active: greedy
        # decoding with these still needs the full path.
        if (
            input_batch.all_greedy is True
            and generate_params_if_all_greedy is False
            and input_batch.no_penalties
            and not has_bad_words
        ):
            return cls(all_greedy=True, logprobs=needs_logprobs)

        num_reqs = input_batch.num_reqs

        def fill_slice(cpu_tensor: torch.Tensor, fill_val) -> torch.Tensor:
            # Pad value is the default one.
            cpu_tensor[num_reqs:padded_num_reqs] = fill_val

        fill_slice(
            input_batch.temperature_cpu_tensor, DEFAULT_SAMPLING_PARAMS["temperature"]
        )
        fill_slice(input_batch.min_p_cpu_tensor, DEFAULT_SAMPLING_PARAMS["min_p"])
        fill_slice(input_batch.top_k_cpu_tensor, DEFAULT_SAMPLING_PARAMS["top_k"])
        fill_slice(input_batch.top_p_cpu_tensor, DEFAULT_SAMPLING_PARAMS["top_p"])

        has_penalties = not input_batch.no_penalties
        output_token_counts = None
        prompt_token_mask_t = None
        presence_penalties_t = None
        frequency_penalties_t = None
        repetition_penalties_t = None
        if has_penalties and vocab_size is not None:
            fill_slice(
                input_batch.presence_penalties_cpu_tensor,
                DEFAULT_SAMPLING_PARAMS["presence_penalties"],
            )
            fill_slice(
                input_batch.frequency_penalties_cpu_tensor,
                DEFAULT_SAMPLING_PARAMS["frequency_penalties"],
            )
            fill_slice(
                input_batch.repetition_penalties_cpu_tensor,
                DEFAULT_SAMPLING_PARAMS["repetition_penalties"],
            )
            presence_penalties_t = input_batch.presence_penalties_cpu_tensor[
                :padded_num_reqs
            ].to(xla_device)
            frequency_penalties_t = input_batch.frequency_penalties_cpu_tensor[
                :padded_num_reqs
            ].to(xla_device)
            repetition_penalties_t = input_batch.repetition_penalties_cpu_tensor[
                :padded_num_reqs
            ].to(xla_device)
            output_token_counts = cls._compute_token_counts(
                input_batch.req_output_token_ids,
                padded_num_reqs,
                vocab_size,
            ).to(xla_device)
            prompt_token_mask_t = cls._compute_prompt_mask(
                input_batch.num_prompt_tokens,
                input_batch.token_ids_cpu,
                padded_num_reqs,
                vocab_size,
            ).to(xla_device)

        bad_words_mask_t = None
        if has_bad_words and vocab_size is not None:
            bad_words_mask_t = cls._compute_bad_words_mask(
                input_batch.bad_words_token_ids,
                input_batch.req_output_token_ids,
                padded_num_reqs,
                vocab_size,
            ).to(xla_device)

        # Slice persistent device tensors to a fixed pre-compiled padded shape.
        return cls(
            temperature=input_batch.temperature_cpu_tensor[:padded_num_reqs].to(
                xla_device
            ),
            all_greedy=input_batch.all_greedy,
            all_random=input_batch.all_random,
            top_p=input_batch.top_p_cpu_tensor[:padded_num_reqs].to(xla_device),
            top_k=input_batch.top_k_cpu_tensor[:padded_num_reqs].to(xla_device),
            min_p=input_batch.min_p_cpu_tensor[:padded_num_reqs].to(xla_device),
            logprobs=needs_logprobs,
            no_penalties=not has_penalties,
            output_token_counts=output_token_counts,
            prompt_token_mask=prompt_token_mask_t,
            presence_penalties=presence_penalties_t,
            frequency_penalties=frequency_penalties_t,
            repetition_penalties=repetition_penalties_t,
            no_bad_words=not has_bad_words,
            bad_words_mask=bad_words_mask_t,
        )
