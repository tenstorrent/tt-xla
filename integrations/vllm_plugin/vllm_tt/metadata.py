# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass, field
from typing import Optional

import torch

from .input_batch import InputBatch

DEFAULT_SAMPLING_PARAMS = dict(
    temperature=-1.0,
    min_p=0.0,
    # strictly disabled for now
    top_k=0,
    top_p=1.0,
    # frequency_penalties=0.0,
    # presence_penalties=0.0,
    # repetition_penalties=0.0,
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

    # TODO No penalties for now
    no_penalties: bool = True
    prompt_token_ids = None
    frequency_penalties = None
    presence_penalties = None
    repetition_penalties = None
    # should use tensor
    output_token_ids: list[list[int]] = field(default_factory=lambda: list())

    min_tokens = None  # impl is not vectorized

    no_logit_bias: bool = True
    logit_bias_tensor: Optional[torch.Tensor] = None

    no_bad_words: bool = True
    bad_words_mask: Optional[torch.Tensor] = None

    allowed_token_ids_mask = None

    # Generator not supported by xla
    _generators: dict[int, torch.Generator] = field(default_factory=lambda: dict())

    @property
    def generators(self) -> dict[int, torch.Generator]:
        # Generator not supported by torch/xla. This field must be immutable.
        return self._generators

    @classmethod
    def from_input_batch(
        cls,
        input_batch: InputBatch,
        padded_num_reqs: int,
        xla_device: torch.device,
        generate_params_if_all_greedy: bool = False,
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

        num_reqs = input_batch.num_reqs

        # Build logit_bias tensor before early return (needed even for greedy).
        has_logit_bias = any(b is not None for b in input_batch.logit_bias[:num_reqs])
        if has_logit_bias:
            logit_bias_cpu = torch.zeros(
                padded_num_reqs, input_batch.vocab_size, dtype=torch.float32
            )
            for req_idx, bias_dict in enumerate(input_batch.logit_bias[:num_reqs]):
                if bias_dict is not None:
                    for token_id, bias_val in bias_dict.items():
                        logit_bias_cpu[req_idx, token_id] = bias_val
            logit_bias_tensor = logit_bias_cpu.to(xla_device)
            no_logit_bias = False
        else:
            logit_bias_tensor = None
            no_logit_bias = True

        # Build bad_words_mask tensor (single-token bad words only).
        has_bad_words = bool(input_batch.bad_words_token_ids)
        if has_bad_words:
            bad_words_cpu = torch.zeros(
                padded_num_reqs, input_batch.vocab_size, dtype=torch.float32
            )
            for req_idx, token_seqs in input_batch.bad_words_token_ids.items():
                for token_seq in token_seqs:
                    if len(token_seq) == 1:
                        bad_words_cpu[req_idx, token_seq[0]] = float("-inf")
            bad_words_mask = bad_words_cpu.to(xla_device)
            no_bad_words = False
        else:
            bad_words_mask = None
            no_bad_words = True

        # Early return to avoid unnecessary cpu to tpu copy
        if input_batch.all_greedy is True and generate_params_if_all_greedy is False:
            return cls(
                all_greedy=True,
                logprobs=needs_logprobs,
                no_logit_bias=no_logit_bias,
                logit_bias_tensor=logit_bias_tensor,
                no_bad_words=no_bad_words,
                bad_words_mask=bad_words_mask,
            )

        def fill_slice(cpu_tensor: torch.Tensor, fill_val) -> torch.Tensor:
            # Pad value is the default one.
            cpu_tensor[num_reqs:padded_num_reqs] = fill_val

        fill_slice(
            input_batch.temperature_cpu_tensor, DEFAULT_SAMPLING_PARAMS["temperature"]
        )
        fill_slice(input_batch.min_p_cpu_tensor, DEFAULT_SAMPLING_PARAMS["min_p"])
        fill_slice(input_batch.top_k_cpu_tensor, DEFAULT_SAMPLING_PARAMS["top_k"])
        fill_slice(input_batch.top_p_cpu_tensor, DEFAULT_SAMPLING_PARAMS["top_p"])

        # Slice persistent device tensors to a fixed pre-compiled padded shape.
        return cls(
            temperature=input_batch.temperature_cpu_tensor[:padded_num_reqs].to(
                xla_device
            ),
            all_greedy=input_batch.all_greedy,
            all_random=input_batch.all_random,
            # TODO enable more and avoid returning None values
            top_p=input_batch.top_p_cpu_tensor[:padded_num_reqs].to(xla_device),
            top_k=input_batch.top_k_cpu_tensor[:padded_num_reqs].to(xla_device),
            min_p=input_batch.min_p_cpu_tensor[:padded_num_reqs].to(xla_device),
            logprobs=needs_logprobs,
            no_logit_bias=no_logit_bias,
            logit_bias_tensor=logit_bias_tensor,
            no_bad_words=no_bad_words,
            bad_words_mask=bad_words_mask,
        )
