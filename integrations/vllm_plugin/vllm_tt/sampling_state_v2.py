# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""TT ``SamplingStates`` for the vLLM v2 model runner.

A host-side sampling-param table keyed by ``req_state_idx``; sampling itself runs
through ``XLASupportedSamplingMetadata`` (metadata.py). Each step
``make_batch_view`` gathers a batch-ordered padded view for ``from_v2_states``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

import numpy as np
import torch
from vllm.sampling_params import SamplingType
from vllm.v1.sample.logits_processor import LogitsProcessors

from .metadata import DEFAULT_SAMPLING_PARAMS

if TYPE_CHECKING:
    from vllm.sampling_params import SamplingParams

    from .request_state import TTRequestState


@dataclass
class SamplingBatchView:
    """Padded, batch-ordered view exposing exactly what ``from_input_batch`` reads."""

    num_reqs: int
    vocab_size: int
    temperature_cpu_tensor: torch.Tensor
    top_p_cpu_tensor: torch.Tensor
    top_k_cpu_tensor: torch.Tensor
    min_p_cpu_tensor: torch.Tensor
    presence_penalties_cpu_tensor: torch.Tensor
    frequency_penalties_cpu_tensor: torch.Tensor
    repetition_penalties_cpu_tensor: torch.Tensor
    all_greedy: bool
    all_random: bool
    no_penalties: bool
    logit_bias: list[Optional[dict[int, float]]]
    bad_words_token_ids: dict[int, list[list[int]]]
    min_tokens: dict[int, tuple[int, set[int]]]
    generators: dict[int, torch.Generator]
    no_allowed_token_ids: bool
    allowed_token_ids_mask_cpu_tensor: Optional[torch.Tensor]
    req_output_token_ids: list[Optional[list[int]]]
    num_prompt_tokens: np.ndarray
    token_ids_cpu: np.ndarray
    max_num_logprobs: int
    # Always False: only pooling models need token ids here.
    logits_processing_needs_token_ids: np.ndarray
    # This step's drafts per row; penalties and bad-words expand output tokens
    # over them, one logits row per draft.
    spec_token_ids: list[list[int]]
    # Neither runner builds custom logits processors.
    logitsprocs: LogitsProcessors
    logitsprocs_need_output_token_ids: bool


class TTSamplingStates:
    """Persistent per-slot sampling-param table for the TT MRv2 runner."""

    def __init__(self, max_num_reqs: int, vocab_size: int):
        self.max_num_reqs = max_num_reqs
        self.vocab_size = vocab_size

        self.temperature = np.full(
            max_num_reqs, DEFAULT_SAMPLING_PARAMS["temperature"], dtype=np.float32
        )
        self.top_p = np.full(
            max_num_reqs, DEFAULT_SAMPLING_PARAMS["top_p"], dtype=np.float32
        )
        self.top_k = np.full(
            max_num_reqs, DEFAULT_SAMPLING_PARAMS["top_k"], dtype=np.int32
        )
        self.min_p = np.full(
            max_num_reqs, DEFAULT_SAMPLING_PARAMS["min_p"], dtype=np.float32
        )
        self.frequency_penalties = np.full(
            max_num_reqs,
            DEFAULT_SAMPLING_PARAMS["frequency_penalties"],
            dtype=np.float32,
        )
        self.presence_penalties = np.full(
            max_num_reqs,
            DEFAULT_SAMPLING_PARAMS["presence_penalties"],
            dtype=np.float32,
        )
        self.repetition_penalties = np.full(
            max_num_reqs,
            DEFAULT_SAMPLING_PARAMS["repetition_penalties"],
            dtype=np.float32,
        )
        # Default (empty) slots are greedy, matching the temperature sentinel.
        self.is_greedy = np.ones(max_num_reqs, dtype=np.bool_)

        self.min_tokens: dict[int, tuple[int, set[int]]] = {}
        self.generators: dict[int, torch.Generator] = {}
        self.logit_bias: list[Optional[dict[int, float]]] = [None] * max_num_reqs
        self.bad_words_token_ids: dict[int, list[list[int]]] = {}
        self.allowed_token_ids: dict[int, list[int]] = {}
        self.num_logprobs: dict[int, int] = {}

    def add_request(
        self,
        slot: int,
        sampling_params: "SamplingParams",
        generator: "torch.Generator | None" = None,
    ) -> None:
        if sampling_params.sampling_type == SamplingType.GREEDY:
            # Zero temperature avoids a divide-by-zero in apply_temperature.
            self.temperature[slot] = 0.0
            self.is_greedy[slot] = True
        else:
            self.temperature[slot] = sampling_params.temperature
            self.is_greedy[slot] = False

        self.top_p[slot] = sampling_params.top_p
        top_k = sampling_params.top_k
        if not 0 < top_k < self.vocab_size:
            top_k = self.vocab_size
        self.top_k[slot] = top_k
        self.min_p[slot] = sampling_params.min_p
        self.frequency_penalties[slot] = sampling_params.frequency_penalty
        self.presence_penalties[slot] = sampling_params.presence_penalty
        self.repetition_penalties[slot] = sampling_params.repetition_penalty

        if sampling_params.min_tokens:
            self.min_tokens[slot] = (
                sampling_params.min_tokens,
                sampling_params.all_stop_token_ids,
            )
        # Only seeded requests carry a generator; the rest use the global RNG.
        if generator is not None:
            self.generators[slot] = generator
        if sampling_params.logprobs is not None:
            self.num_logprobs[slot] = sampling_params.logprobs
        if sampling_params.logit_bias is not None:
            self.logit_bias[slot] = sampling_params.logit_bias
        if sampling_params.allowed_token_ids:
            self.allowed_token_ids[slot] = sampling_params.allowed_token_ids
        if sampling_params.bad_words_token_ids:
            self.bad_words_token_ids[slot] = sampling_params.bad_words_token_ids

    def remove_request(self, slot: int) -> None:
        self.temperature[slot] = DEFAULT_SAMPLING_PARAMS["temperature"]
        self.top_p[slot] = DEFAULT_SAMPLING_PARAMS["top_p"]
        self.top_k[slot] = DEFAULT_SAMPLING_PARAMS["top_k"]
        self.min_p[slot] = DEFAULT_SAMPLING_PARAMS["min_p"]
        self.frequency_penalties[slot] = DEFAULT_SAMPLING_PARAMS["frequency_penalties"]
        self.presence_penalties[slot] = DEFAULT_SAMPLING_PARAMS["presence_penalties"]
        self.repetition_penalties[slot] = DEFAULT_SAMPLING_PARAMS[
            "repetition_penalties"
        ]
        self.is_greedy[slot] = True
        self.min_tokens.pop(slot, None)
        self.generators.pop(slot, None)
        self.logit_bias[slot] = None
        self.bad_words_token_ids.pop(slot, None)
        self.allowed_token_ids.pop(slot, None)
        self.num_logprobs.pop(slot, None)

    def make_batch_view(
        self,
        request_state: "TTRequestState",
        input_batch: Any,
        padded_num_reqs: int,
    ) -> SamplingBatchView:
        """Gather a batch-ordered, padded view for the current step.

        ``input_batch`` is any object exposing ``num_reqs`` + ``idx_mapping_np``;
        the runner passes a per-pass namespace, not a vLLM InputBatch. Rows
        [num_reqs:padded] carry the sampler defaults.
        """
        num_reqs = input_batch.num_reqs
        slots = [int(input_batch.idx_mapping_np[b]) for b in range(num_reqs)]
        vocab = self.vocab_size

        def _full(name):
            return torch.full(
                (padded_num_reqs,),
                float(DEFAULT_SAMPLING_PARAMS[name]),
                dtype=torch.float32,
            )

        temperature = _full("temperature")
        top_p = _full("top_p")
        min_p = _full("min_p")
        presence = _full("presence_penalties")
        frequency = _full("frequency_penalties")
        repetition = _full("repetition_penalties")
        top_k = torch.full(
            (padded_num_reqs,),
            int(DEFAULT_SAMPLING_PARAMS["top_k"]),
            dtype=torch.int32,
        )

        max_model_len = request_state.all_token_ids.shape[1]
        token_ids_cpu = np.zeros((padded_num_reqs, max_model_len), dtype=np.int32)
        num_prompt_tokens = np.zeros(padded_num_reqs, dtype=np.int32)
        req_output_token_ids: list[Optional[list[int]]] = [None] * padded_num_reqs

        logit_bias: list[Optional[dict[int, float]]] = [None] * padded_num_reqs
        spec_token_ids: list[list[int]] = [[] for _ in range(padded_num_reqs)]
        bad_words: dict[int, list[list[int]]] = {}
        min_tokens: dict[int, tuple[int, set[int]]] = {}
        generators: dict[int, torch.Generator] = {}

        has_allowed = any(slot in self.allowed_token_ids for slot in slots)
        allowed_mask = (
            torch.zeros(padded_num_reqs, vocab, dtype=torch.bool)
            if has_allowed
            else None
        )

        num_greedy = 0
        has_penalties = False
        max_num_logprobs = 0

        for b, slot in enumerate(slots):
            temperature[b] = float(self.temperature[slot])
            top_p[b] = float(self.top_p[slot])
            top_k[b] = int(self.top_k[slot])
            min_p[b] = float(self.min_p[slot])
            presence[b] = float(self.presence_penalties[slot])
            frequency[b] = float(self.frequency_penalties[slot])
            repetition[b] = float(self.repetition_penalties[slot])

            if bool(self.is_greedy[slot]):
                num_greedy += 1
            if (
                self.frequency_penalties[slot] != 0.0
                or self.presence_penalties[slot] != 0.0
                or self.repetition_penalties[slot] != 1.0
            ):
                has_penalties = True

            prompt_len = int(request_state.prompt_len[slot])
            total_len = int(request_state.total_len[slot])
            token_ids_cpu[b] = request_state.all_token_ids[slot]
            num_prompt_tokens[b] = prompt_len
            req_output_token_ids[b] = request_state.all_token_ids[
                slot, prompt_len:total_len
            ].tolist()

            num_drafts = int(request_state.num_draft_tokens[slot])
            if num_drafts:
                spec_token_ids[b] = request_state.draft_tokens[
                    slot, :num_drafts
                ].tolist()

            logit_bias[b] = self.logit_bias[slot]
            if slot in self.bad_words_token_ids:
                bad_words[b] = self.bad_words_token_ids[slot]
            if slot in self.min_tokens:
                min_tokens[b] = self.min_tokens[slot]
            if slot in self.generators:
                generators[b] = self.generators[slot]
            if slot in self.num_logprobs:
                max_num_logprobs = max(max_num_logprobs, self.num_logprobs[slot])
            if allowed_mask is not None and slot in self.allowed_token_ids:
                allowed_mask[b, :] = True  # True == disallowed
                allowed = [t for t in self.allowed_token_ids[slot] if t < vocab]
                if allowed:
                    allowed_mask[b, allowed] = False

        return SamplingBatchView(
            num_reqs=num_reqs,
            vocab_size=vocab,
            temperature_cpu_tensor=temperature,
            top_p_cpu_tensor=top_p,
            top_k_cpu_tensor=top_k,
            min_p_cpu_tensor=min_p,
            presence_penalties_cpu_tensor=presence,
            frequency_penalties_cpu_tensor=frequency,
            repetition_penalties_cpu_tensor=repetition,
            all_greedy=num_reqs > 0 and num_greedy == num_reqs,
            all_random=num_reqs > 0 and num_greedy == 0,
            no_penalties=not has_penalties,
            logit_bias=logit_bias,
            bad_words_token_ids=bad_words,
            min_tokens=min_tokens,
            generators=generators,
            no_allowed_token_ids=not has_allowed,
            allowed_token_ids_mask_cpu_tensor=allowed_mask,
            req_output_token_ids=req_output_token_ids,
            num_prompt_tokens=num_prompt_tokens,
            token_ids_cpu=token_ids_cpu,
            max_num_logprobs=max_num_logprobs,
            logits_processing_needs_token_ids=np.zeros(padded_num_reqs, dtype=bool),
            spec_token_ids=spec_token_ids,
            logitsprocs=LogitsProcessors(),
            logitsprocs_need_output_token_ids=False,
        )
