# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sampler layer implementing XLA supported operations."""

import torch
import torch.nn as nn
from vllm.v1.outputs import LogprobsTensors, SamplerOutput

from .metadata import XLASupportedSamplingMetadata

_SAMPLING_EPS = 1e-5


def count_tokens_ge(logprobs: torch.Tensor, threshold: torch.Tensor) -> torch.Tensor:
    """Count tokens per row whose logprob >= threshold.

    Returns int64 (natural sum dtype).  Callers that need int32 — e.g.
    gather_logprobs for the LogprobsTensors convention — must cast after.
    """
    return (logprobs >= threshold).sum(-1)


class Sampler(nn.Module):
    def __init__(self):
        # TODO(houseroad): Add support for logprobs_mode (a future extension
        # for configuring which logprobs are returned).  Basic logprob support
        # is already working: when logprobs are requested, model_runner.py
        # calls gather_logprobs() after forward() and passes LogprobsLists
        # directly to the engine.  logprobs_tensors is intentionally None in
        # forward() — see the comment there.
        super().__init__()

    def forward(
        self,
        logits: torch.Tensor,
        sampling_metadata: XLASupportedSamplingMetadata,
    ) -> SamplerOutput:
        # Use float32 for the logits.
        logits = logits.to(torch.float32)
        # Sample the next token.
        sampled = self.sample(logits, sampling_metadata)

        # These are XLA tensors.
        sampler_output = SamplerOutput(
            # The sampled tokens are expanded to 2D tensor with shape
            # [num_requests, 1], where each row represents one generated
            # token per request.
            sampled_token_ids=sampled.unsqueeze(-1),
            # Logprobs do not flow through SamplerOutput.  When logprobs are
            # requested, model_runner.py calls gather_logprobs() after
            # forward() and assembles LogprobsLists directly — bypassing this
            # field entirely.  Setting logprobs_tensors=None here is
            # intentional.
            logprobs_tensors=None,
        )
        return sampler_output

    def apply_temperature(
        self,
        logits: torch.Tensor,
        temp: torch.Tensor,
        all_random: bool = False,
    ) -> torch.Tensor:
        # Avoid division by zero for greedy sampling (temperature ~ 0.0).
        if not all_random:
            temp = torch.where(temp < _SAMPLING_EPS, 1.0, temp)
        return logits.div_(temp.unsqueeze(dim=1))

    def greedy_sample(self, logits: torch.Tensor) -> torch.Tensor:
        return logits.argmax(dim=-1).view(-1)

    def apply_penalties(
        self,
        logits: torch.Tensor,
        output_token_counts: torch.Tensor,
        prompt_token_mask: torch.Tensor,
        presence_penalties: torch.Tensor,
        frequency_penalties: torch.Tensor,
        repetition_penalties: torch.Tensor,
    ) -> torch.Tensor:
        """Apply presence, frequency, and repetition penalties on-device.

        Uses element-wise masking ops only — no sort, scatter, or gather.
        Matches the vLLM GPU spec (vllm/model_executor/layers/utils.py):
          - repetition_penalty: applied to tokens in prompt ∪ output
          - frequency_penalty:  applied to output tokens, scaled by count
          - presence_penalty:   applied to output tokens, flat per token

        ``output_token_counts`` is a [batch, vocab] float32 count tensor.
        ``prompt_token_mask`` is a [batch, vocab] bool mask for prompt tokens.
        Both are pre-built on CPU and transferred to device before graph execution.
        """
        occurred_output = output_token_counts > 0  # [batch, vocab] bool

        # Apply in the same order as the GPU reference:
        # repetition -> frequency -> presence.

        # Repetition penalty covers prompt ∪ output tokens.
        rep_mask = occurred_output | prompt_token_mask
        rep = repetition_penalties.unsqueeze(1)  # [batch, 1]
        penalty_factor = torch.where(logits > 0, torch.reciprocal(rep), rep)
        logits = torch.where(rep_mask, logits * penalty_factor, logits)

        # Frequency penalty: subtract penalty scaled by output occurrence count.
        logits -= frequency_penalties.unsqueeze(1) * output_token_counts.to(
            logits.dtype
        )

        # Presence penalty: subtract a flat penalty for each token that appeared in output.
        logits -= presence_penalties.unsqueeze(1) * occurred_output.to(logits.dtype)

        return logits

    def apply_bad_words(
        self,
        logits: torch.Tensor,
        bad_words_mask: torch.Tensor,
    ) -> torch.Tensor:
        return logits + bad_words_mask

    def apply_logit_bias(
        self,
        logits: torch.Tensor,
        logit_bias_tensor: torch.Tensor,
    ) -> torch.Tensor:
        return logits + logit_bias_tensor

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: XLASupportedSamplingMetadata,
    ) -> torch.Tensor:
        # Apply bad_words mask first (sets banned tokens to -inf).
        if not sampling_metadata.no_bad_words:
            logits = self.apply_bad_words(logits, sampling_metadata.bad_words_mask)

        # Apply logit_bias before computing greedy argmax.
        if not sampling_metadata.no_logit_bias:
            logits = self.apply_logit_bias(logits, sampling_metadata.logit_bias_tensor)

        assert sampling_metadata.temperature is not None

        # Apply penalties before temperature so both greedy and random paths
        # see penalized logits.
        if not sampling_metadata.no_penalties:
            logits = self.apply_penalties(
                logits,
                sampling_metadata.output_token_counts,
                sampling_metadata.prompt_token_mask,
                sampling_metadata.presence_penalties,
                sampling_metadata.frequency_penalties,
                sampling_metadata.repetition_penalties,
            )

        greedy_sampled = self.greedy_sample(logits)

        # Apply temperature.
        logits = self.apply_temperature(
            logits, sampling_metadata.temperature, sampling_metadata.all_random
        )

        # Apply min_p.
        if sampling_metadata.min_p is not None:
            logits = self.apply_min_p(logits, sampling_metadata.min_p)

        # Apply top_k and/or top_p.
        logits = apply_top_k_top_p(
            logits,
            sampling_metadata.top_k,
            sampling_metadata.top_p,
        )

        # Random sample.
        probs = logits.softmax(dim=-1, dtype=torch.float32)
        random_sampled = self.random_sample(probs, sampling_metadata.generators)

        sampled = torch.where(
            sampling_metadata.temperature < _SAMPLING_EPS,
            greedy_sampled,
            random_sampled,
        )
        return sampled

    def compute_logprobs(self, logits: torch.Tensor) -> torch.Tensor:
        return logits.log_softmax(dim=-1, dtype=torch.float32)

    def gather_logprobs(
        self,
        logprobs: torch.Tensor,
        num_logprobs: int,
        token_ids: torch.Tensor,
    ) -> LogprobsTensors:
        """
        Gather logprobs for topk and sampled/prompt token.

        Args:
          logprobs: (num tokens) x (vocab) tensor
          num_logprobs: minimum number of logprobs to
                        retain per token
          token_ids: prompt tokens (if prompt logprobs)
                     or sampled tokens (if sampled
                     logprobs); 1D token ID tensor
                     with (num tokens) elements

        Returns:
          Top-k int indices tensor, (num tokens) x (num_logprobs + 1)
          Top-k float logprobs tensor, (num tokens) x (num_logprobs + 1)
          Sampled token rank tensor, (num tokens)
        """
        # Find the topK values.
        topk_logprobs, topk_indices = torch.topk(logprobs, num_logprobs, dim=-1)

        # Get with the logprob of the prompt or sampled token.
        # Cast to int32: TT does not support int64 as a gather index (returns
        # NaN).  The on-device cast routes through bfloat16 — a known TT hw
        # limitation — so large vocab indices are rounded (e.g. 33042→33024).
        token_ids = token_ids.to(torch.int32).unsqueeze(-1)
        token_logprobs = logprobs.gather(-1, token_ids)

        # Cast to int32 to match LogprobsTensors.empty_cpu() convention.
        token_ranks = count_tokens_ge(logprobs, token_logprobs).to(torch.int32)

        # Concatenate together with the topk.
        # Cast topk_indices to int32 to match token_ids dtype for cat.
        indices = torch.cat((token_ids, topk_indices.to(torch.int32)), dim=1)
        logprobs = torch.cat((token_logprobs, topk_logprobs), dim=1)

        return LogprobsTensors(indices, logprobs, token_ranks)

    def apply_min_p(
        self,
        logits: torch.Tensor,
        min_p: torch.Tensor,
    ) -> torch.Tensor:
        """
        Filters logits using adaptive probability thresholding.
        """
        # Convert logits to probability distribution
        probability_values = torch.nn.functional.softmax(logits, dim=-1)
        # Calculate maximum probabilities per sequence
        max_probabilities = torch.amax(probability_values, dim=-1, keepdim=True)
        # Reshape min_p for broadcasting
        adjusted_min_p = min_p.unsqueeze(1) * max_probabilities
        # Identify valid tokens using threshold comparison
        valid_token_mask = probability_values >= adjusted_min_p
        # Apply mask using boolean indexing (xla friendly)
        logits.masked_fill_(~valid_token_mask, -float("inf"))
        return logits

    def random_sample(
        self,
        probs: torch.Tensor,
        generators: dict[int, torch.Generator],
    ) -> torch.Tensor:
        q = torch.empty_like(probs)
        # NOTE(woosuk): To batch-process the requests without their own seeds,
        # which is the common case, we first assume that every request does
        # not have its own seed. Then, we overwrite the values for the requests
        # that have their own seeds.
        q.exponential_()
        if generators:
            for i, generator in generators.items():
                q[i].exponential_(generator=generator)
        return probs.div_(q).argmax(dim=-1).view(-1)


def apply_top_k_top_p(
    logits: torch.Tensor,
    k: torch.Tensor | None,
    p: torch.Tensor | None,
) -> torch.Tensor:
    """
    Apply top-k and top-p optimized for TPU.

    This algorithm avoids using torch.scatter which is extremely slow on TPU.
    This is achieved by finding a "cut-off" element in the original logit, and
    after thresholding the logit using this cut-off, the remaining elements
    shall constitute the top-p set.

    Note: in the case of tie (i.e. multiple cut-off elements present in the
    logit), all tie elements are included in the top-p set. In other words,
    this function does not break ties. Instead, these tie tokens have equal
    chance of being chosen during final sampling, so we can consider the tie
    being broken then.
    """
    probs = logits.softmax(dim=-1)
    probs_sort, _ = probs.sort(dim=-1, descending=False)

    if k is not None:
        top_k_count = probs_sort.size(1) - k.to(torch.long)  # shape: (batch, )
        top_k_count = top_k_count.unsqueeze(dim=1)
        top_k_cutoff = probs_sort.gather(-1, top_k_count)

        # Make sure the no top-k rows are no-op.
        no_top_k_mask = (k == logits.shape[1]).unsqueeze(dim=1)
        top_k_cutoff.masked_fill_(no_top_k_mask, -float("inf"))

        elements_to_discard = probs < top_k_cutoff
        logits.masked_fill_(elements_to_discard, -float("inf"))

    if p is not None:
        cumprob = torch.cumsum(probs_sort, dim=-1)
        top_p_mask = cumprob <= 1 - p.unsqueeze(dim=1)
        top_p_mask[:, -1] = False  # at least one

        top_p_count = top_p_mask.sum(dim=-1).unsqueeze(1)
        top_p_cutoff = probs_sort.gather(-1, top_p_count)
        elements_to_discard = probs < top_p_cutoff
        logits.masked_fill_(elements_to_discard, -float("inf"))

    return logits
