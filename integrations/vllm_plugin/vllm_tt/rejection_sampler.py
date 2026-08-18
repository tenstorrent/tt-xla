# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Portions (c) 2026 Tenstorrent AI ULC

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

import torch
import torch.nn as nn
from vllm.v1.outputs import LogprobsLists, LogprobsTensors, SamplerOutput
from vllm.v1.sample.logits_processor.builtin import MinTokensLogitsProcessor
from vllm.v1.sample.ops.bad_words import apply_bad_words_with_drafts
from vllm.v1.sample.ops.penalties import apply_all_penalties
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata

_SAMPLING_EPS = 1e-5
_PLACEHOLDER_TOKEN_ID = -1


class RejectionSampler(nn.Module):
    """TT-compatible speculative rejection sampler.

    This mirrors the public surface of vLLM's RejectionSampler but avoids the
    Triton kernels used in its forward path. It is intended for TT runs where
    speculative decoding currently uses ngram proposals (draft_probs=None).
    """

    def __init__(self, sampler: nn.Module):
        super().__init__()
        self.sampler = sampler

    def forward(
        self,
        metadata: SpecDecodeMetadata,
        draft_probs: torch.Tensor | None,
        logits: torch.Tensor,
        sampling_metadata,
    ) -> SamplerOutput:
        assert logits is not None
        if draft_probs is not None:
            raise NotImplementedError(
                "TT RejectionSampler currently supports only draft_probs=None "
                "(ngram speculative decoding)."
            )

        bonus_logits = logits[metadata.bonus_logits_indices]
        bonus_sampler_output = self.sampler(
            logits=bonus_logits,
            sampling_metadata=replace(
                sampling_metadata,
                max_num_logprobs=-1,
            ),
        )
        # Transfer first, cast on host. An on-device .to(dtype) before .cpu()
        # makes the tt-mlir runtime memcpy assert on a dst/src dtype mismatch
        # (runtime.cpp: dstDataType == getTensorDataType(src)) and abort.
        bonus_token_ids = bonus_sampler_output.sampled_token_ids.cpu().to(torch.int32)

        target_logits = logits[metadata.target_logits_indices].cpu().to(torch.float32)
        target_logits = self.apply_logits_processors(
            target_logits,
            sampling_metadata,
            metadata,
        )

        output_token_ids = self._rejection_sample_fallback(
            draft_token_ids=metadata.draft_token_ids,
            num_draft_tokens=metadata.num_draft_tokens,
            max_spec_len=metadata.max_spec_len,
            cu_num_draft_tokens=metadata.cu_num_draft_tokens,
            target_logits=target_logits,
            bonus_token_ids=bonus_token_ids,
            sampling_metadata=sampling_metadata,
        )

        return SamplerOutput(
            sampled_token_ids=output_token_ids,
            logprobs_tensors=None,
        )

    @staticmethod
    def parse_output(
        output_token_ids: torch.Tensor,
        vocab_size: int,
        discard_req_indices: Sequence[int] = (),
        logprobs_tensors: LogprobsTensors | None = None,
    ) -> tuple[list[list[int]], LogprobsLists | None]:
        output_token_ids_np = output_token_ids.cpu().numpy()
        valid_mask = (output_token_ids_np != _PLACEHOLDER_TOKEN_ID) & (
            output_token_ids_np < vocab_size
        )
        output_logprobs = None
        if logprobs_tensors is not None:
            cu_num_tokens = [0] + valid_mask.sum(axis=1).cumsum().tolist()
            filtered_tensors = logprobs_tensors.filter(valid_mask.flatten())
            output_logprobs = filtered_tensors.tolists(cu_num_tokens)

        if len(discard_req_indices) > 0:
            valid_mask[list(discard_req_indices)] = False
        outputs = [
            row[valid_mask[i]].tolist() for i, row in enumerate(output_token_ids_np)
        ]
        return outputs, output_logprobs

    def apply_logits_processors(
        self,
        logits: torch.Tensor,
        sampling_metadata,
        metadata: SpecDecodeMetadata,
    ) -> torch.Tensor:
        has_penalties = not sampling_metadata.no_penalties
        any_penalties_or_bad_words = (
            sampling_metadata.bad_words_token_ids or has_penalties
        )

        output_token_ids = sampling_metadata.output_token_ids
        if any_penalties_or_bad_words:
            output_token_ids = self._combine_outputs_with_spec_tokens(
                output_token_ids,
                sampling_metadata.spec_token_ids,
            )

        if sampling_metadata.allowed_token_ids_mask is not None or has_penalties:
            num_requests = len(metadata.num_draft_tokens)
            num_draft_tokens = torch.tensor(metadata.num_draft_tokens, device="cpu")
            original_indices = torch.arange(num_requests, device="cpu")
            repeat_indices_cpu = original_indices.repeat_interleave(num_draft_tokens)
            repeat_indices = repeat_indices_cpu.to(device=logits.device)
            logits = self.apply_penalties(
                logits,
                sampling_metadata,
                metadata,
                repeat_indices,
                output_token_ids,
            )

            if sampling_metadata.allowed_token_ids_mask is not None:
                token_mask = sampling_metadata.allowed_token_ids_mask[repeat_indices]
                logits.masked_fill_(token_mask, float("-inf"))

        if bad_words_token_ids := sampling_metadata.bad_words_token_ids:
            apply_bad_words_with_drafts(
                logits, bad_words_token_ids, output_token_ids, metadata.num_draft_tokens
            )

        for processor in sampling_metadata.logitsprocs.non_argmax_invariant:
            if isinstance(processor, MinTokensLogitsProcessor):
                logits = processor.apply_with_spec_decode(
                    logits, metadata.num_draft_tokens
                )

        return logits

    @staticmethod
    def apply_penalties(
        logits: torch.Tensor,
        sampling_metadata,
        metadata: SpecDecodeMetadata,
        repeat_indices: torch.Tensor,
        output_token_ids: list[list[int]],
    ) -> torch.Tensor:
        if sampling_metadata.no_penalties:
            return logits

        assert sampling_metadata.prompt_token_ids is not None

        prompt_token_ids = sampling_metadata.prompt_token_ids[repeat_indices]
        presence_penalties = sampling_metadata.presence_penalties[repeat_indices]
        frequency_penalties = sampling_metadata.frequency_penalties[repeat_indices]
        repetition_penalties = sampling_metadata.repetition_penalties[repeat_indices]

        return apply_all_penalties(
            logits,
            prompt_token_ids,
            presence_penalties,
            frequency_penalties,
            repetition_penalties,
            output_token_ids,
        )

    @staticmethod
    def _combine_outputs_with_spec_tokens(
        output_token_ids: list[list[int]],
        spec_token_ids: list[list[int]] | None = None,
    ) -> list[list[int]]:
        if spec_token_ids is None:
            return output_token_ids

        result = []
        for out, spec in zip(output_token_ids, spec_token_ids):
            if len(spec) == 0:
                continue
            result.append(out)
            for i in range(len(spec) - 1):
                result.append([*result[-1], spec[i]])
        return result

    @staticmethod
    def _apply_row_sampling_constraints(
        logits_row: torch.Tensor,
        is_greedy: bool,
        temperature: float,
        top_k: int | None,
        top_p: float | None,
    ) -> torch.Tensor:
        if not is_greedy:
            temp = 1.0 if temperature < _SAMPLING_EPS else temperature
            logits_row = logits_row / temp

            if top_k is not None and top_k > 0 and top_k < logits_row.shape[0]:
                topk_vals, _ = torch.topk(logits_row, top_k)
                logits_row[logits_row < topk_vals[-1]] = float("-inf")

            if top_p is not None and top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits_row, descending=True)
                probs = torch.softmax(sorted_logits, dim=-1)
                mask = torch.cumsum(probs, dim=-1) - probs >= top_p
                sorted_logits[mask] = float("-inf")
                logits_row.scatter_(0, sorted_indices, sorted_logits)

        return logits_row

    def _rejection_sample_fallback(
        self,
        draft_token_ids: torch.Tensor,
        num_draft_tokens: list[int],
        max_spec_len: int,
        cu_num_draft_tokens: torch.Tensor,
        target_logits: torch.Tensor,
        bonus_token_ids: torch.Tensor,
        sampling_metadata,
    ) -> torch.Tensor:
        batch_size = len(num_draft_tokens)
        # Transfer-then-cast, as above (these may still be device tensors).
        draft_token_ids = draft_token_ids.cpu().to(torch.int32)
        target_logits = target_logits.cpu().to(torch.float32)
        bonus_token_ids = bonus_token_ids.cpu().to(torch.int32)
        output_token_ids = torch.full(
            (batch_size, max_spec_len + 1),
            _PLACEHOLDER_TOKEN_ID,
            dtype=torch.int32,
            device="cpu",
        )

        cu = cu_num_draft_tokens.cpu().to(torch.int64).tolist()

        for req_idx in range(batch_size):
            draft_len = int(num_draft_tokens[req_idx])
            req_is_greedy = True
            req_temperature = 1.0
            req_top_k = None
            req_top_p = None
            req_generator = None

            if sampling_metadata is not None:
                if sampling_metadata.temperature is not None:
                    req_temperature = float(
                        sampling_metadata.temperature[req_idx].item()
                    )
                req_is_greedy = sampling_metadata.all_greedy or (
                    req_temperature < _SAMPLING_EPS
                )
                if sampling_metadata.top_k is not None:
                    req_top_k = int(sampling_metadata.top_k[req_idx].item())
                if sampling_metadata.top_p is not None:
                    req_top_p = float(sampling_metadata.top_p[req_idx].item())
                req_generator = sampling_metadata.generators.get(req_idx)

            if draft_len == 0:
                output_token_ids[req_idx, 0] = bonus_token_ids[req_idx, 0]
                continue

            start = 0 if req_idx == 0 else int(cu[req_idx - 1])
            all_accepted = True
            for draft_pos in range(draft_len):
                flat_idx = start + draft_pos
                draft_tid = draft_token_ids[flat_idx]
                constrained_logits = self._apply_row_sampling_constraints(
                    target_logits[flat_idx].clone(),
                    req_is_greedy,
                    req_temperature,
                    req_top_k,
                    req_top_p,
                )

                if req_is_greedy:
                    target_tid = torch.argmax(constrained_logits).to(torch.int32)
                    if draft_tid == target_tid:
                        output_token_ids[req_idx, draft_pos] = draft_tid
                    else:
                        output_token_ids[req_idx, draft_pos] = target_tid
                        all_accepted = False
                        break
                else:
                    probs = torch.softmax(constrained_logits, dim=-1)
                    accept_prob = float(probs[draft_tid].item())
                    sample = torch.rand((), generator=req_generator)
                    if float(sample.item()) <= accept_prob:
                        output_token_ids[req_idx, draft_pos] = draft_tid
                    else:
                        recovered_probs = probs.clone()
                        recovered_probs[draft_tid] = 0.0
                        total = recovered_probs.sum()
                        if float(total.item()) <= 0.0:
                            recovered_tid = torch.argmax(constrained_logits).to(
                                torch.int32
                            )
                        else:
                            recovered_probs /= total
                            recovered_tid = torch.multinomial(
                                recovered_probs,
                                num_samples=1,
                                generator=req_generator,
                            ).to(torch.int32)[0]
                        output_token_ids[req_idx, draft_pos] = recovered_tid
                        all_accepted = False
                        break

            if all_accepted:
                output_token_ids[req_idx, draft_len] = bonus_token_ids[req_idx, 0]

        return output_token_ids
