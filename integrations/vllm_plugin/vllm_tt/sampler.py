# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sampler layer implementing XLA supported operations."""

import math
import os

import torch
import torch.nn as nn
from vllm.v1.outputs import LogprobsTensors, SamplerOutput

from .metadata import XLASupportedSamplingMetadata

# Multi-core topk parameters.
# ttnn.topk uses a multi-core bitonic sort when input dim is a power of 2 and
# < 65536.  We split the vocab into chunks of at most this size, pad each
# chunk to the next power of 2, and run topk per chunk.  All chunk topk results
# are concatenated to form the candidate set for sampling.
_TOPK_MAX_CHUNK_SIZE = 32768  # largest power-of-2 below 65536
_TOPK_K_PER_CHUNK = 32  # candidates kept per chunk


def _next_power_of_2(n: int) -> int:
    return 1 << (n - 1).bit_length()


def _get_topk_split_params(vocab_size: int) -> tuple[int, int]:
    """Return (chunk_size, padded_chunk_size) for chunked topk."""
    num_chunks = math.ceil(vocab_size / _TOPK_MAX_CHUNK_SIZE)
    chunk_size = math.ceil(vocab_size / num_chunks)
    return chunk_size, _next_power_of_2(chunk_size)


_SAMPLING_EPS = 1e-5
_TOPK_BEFORE_ARGMAX = os.environ.get("TT_TOPK_BEFORE_ARGMAX", "") == "1"
_USE_TTNN_SAMPLING = os.environ.get("TT_USE_TTNN_SAMPLING", "1") != "0"
_TTNN_SAMPLING_BATCH_SIZE = 32  # ttnn.sampling kernel requires batch=32


def count_tokens_ge(logprobs: torch.Tensor, threshold: torch.Tensor) -> torch.Tensor:
    """Count tokens per row whose logprob >= threshold, minimum 1.

    Returns a 1-based rank: rank 1 means only the token itself satisfies >=.

    Workaround for https://github.com/tenstorrent/tt-xla/issues/3464:
    tt-metal does not support boolean tensors, so ElementTypeNormalization
    converts i1 (bool) to bfloat16 early in the TTIR pipeline. The
    comparison (logprobs >= threshold) produces bf16 1.0/0.0 values.
    When sum(-1).clamp(min=1) is fused into a single kernel, the result
    is -1 instead of 1 on TT (each op is correct in isolation).
    torch.maximum with an explicit ones tensor avoids the broken fusion.

    Returns int64 (natural sum dtype). Callers that need int32 — e.g.
    gather_logprobs for the LogprobsTensors convention — must cast after.
    """
    counts = (logprobs >= threshold).sum(-1)
    # TODO(#3464): replace with counts.clamp(min=1) once the fused
    # sum+clamp kernel handles bf16-represented booleans correctly.
    return torch.maximum(counts, torch.ones_like(counts))


class Sampler(nn.Module):
    def __init__(self):
        # TODO(houseroad): Add support for logprobs_mode.
        # Note: basic logprob support is already working — when logprobs are
        # requested, model_runner.py calls gather_logprobs() after forward()
        # and passes LogprobsLists directly to the engine.
        # logprobs_tensors is intentionally None in forward() — see comment there.
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
            # Logprobs do not flow through SamplerOutput. When logprobs are
            # requested, model_runner.py calls gather_logprobs() after
            # forward() and assembles LogprobsLists directly — bypassing this
            # field entirely. Setting logprobs_tensors=None here is
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
        # tt-xla#3464: bool tensors are promoted to bf16 by ElementTypeNormalization;
        # keep everything in float32 to avoid mixed-dtype fusion bugs.
        output_token_counts = output_token_counts.to(logits.dtype)
        prompt_float = prompt_token_mask.to(logits.dtype)

        # min(x,1) = x - relu(x-1) for non-negative integer x — avoids
        # clamp(max=...) and bool comparison, both broken on TT (see count_tokens_ge).
        occurred_float = output_token_counts - (output_token_counts - 1).clamp(min=0)
        rep_union = occurred_float + prompt_float
        rep_mask_float = rep_union - (rep_union - 1).clamp(min=0)

        # Repetition penalty: positive logits divided by rep, negative multiplied.
        rep = repetition_penalties.unsqueeze(1)
        pos_logits = logits.clamp(min=0)
        neg_logits = logits.clamp(max=0)
        penalized_logits = pos_logits / rep + neg_logits * rep
        logits = logits + rep_mask_float * (penalized_logits - logits)

        logits -= frequency_penalties.unsqueeze(1) * output_token_counts
        logits -= presence_penalties.unsqueeze(1) * occurred_float

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
        # Apply allowed_token_ids mask (sets disallowed tokens to -inf).
        if not sampling_metadata.no_allowed_token_ids:
            logits = logits + sampling_metadata.allowed_token_ids_mask

        # Apply min_tokens mask (suppresses stop tokens until minimum is reached).
        if not sampling_metadata.no_min_tokens:
            logits = logits + sampling_metadata.min_tokens_mask

        # Apply bad_words mask (sets banned tokens to -inf).
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

        if _USE_TTNN_SAMPLING:
            # ttnn.sampling handles temperature internally (multiplies by
            # 1/temperature), so skip apply_temperature and run topk on
            # raw logits. Pass None for k/p — ttnn.sampling handles them
            # internally. Disable pad-to-32 here: _ttnn_sampling_padded
            # already pads the candidate set to 32 for the kernel, and
            # batched topk on [32, vocab] with -inf dummy rows can return
            # different index sets for row 0 vs [1, vocab] topk.
            filtered_logits, candidate_indices = apply_top_k_top_p_fast(
                logits,
                None,
                None,
                pad_to_batch=False,
            )
            random_sampled_padded = self._ttnn_sampling_padded(
                filtered_logits,
                candidate_indices,
                sampling_metadata,
            )
            pad_batch = _TTNN_SAMPLING_BATCH_SIZE
            batch = logits.shape[0]
            greedy_padded = torch.nn.functional.pad(
                greedy_sampled, (0, pad_batch - batch)
            ).to(torch.int32)
            temp_padded = torch.nn.functional.pad(
                sampling_metadata.temperature, (0, pad_batch - batch), value=1.0
            )
            sampled_padded = torch.where(
                temp_padded < _SAMPLING_EPS,
                greedy_padded,
                random_sampled_padded,
            )
            sampled_padded = sampled_padded.to(torch.int64)
            return sampled_padded[:batch].view(-1)
        else:
            # Apply temperature.
            logits = self.apply_temperature(
                logits, sampling_metadata.temperature, sampling_metadata.all_random
            )

            # Apply min_p.
            if sampling_metadata.min_p is not None:
                logits = self.apply_min_p(logits, sampling_metadata.min_p)

            filtered_logits, candidate_indices = apply_top_k_top_p_fast(
                logits,
                sampling_metadata.top_k,
                sampling_metadata.top_p,
            )

            probs = filtered_logits.softmax(dim=-1, dtype=torch.float32)

            # If seeded sampling, gather q_samples at candidate positions.
            q_samples_reduced = None
            if sampling_metadata.q_samples is not None:
                q_samples_reduced = sampling_metadata.q_samples.gather(
                    1, candidate_indices
                )

            random_sampled_local = self.random_sample(
                probs, sampling_metadata.generators, q_samples_reduced
            )
            # Map local candidate index back to global vocab index.
            random_sampled = candidate_indices.gather(
                1, random_sampled_local.unsqueeze(-1)
            ).squeeze(-1)

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

        # Get the logprob of the prompt or sampled token.
        token_ids = token_ids.unsqueeze(-1)
        token_logprobs = logprobs.gather(-1, token_ids)

        token_ranks = count_tokens_ge(logprobs, token_logprobs)

        # Cast to int32 for LogprobsTensors (vLLM convention).
        indices = torch.cat(
            (token_ids.to(torch.int32), topk_indices.to(torch.int32)),
            dim=1,
        )
        token_ranks = token_ranks.to(torch.int32)
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
        q_samples: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if q_samples is not None:
            q = q_samples
        else:
            q = torch.empty_like(probs)
            # NOTE(woosuk): To batch-process the requests without their own
            # seeds, which is the common case, we first assume that every
            # request does not have its own seed. Then, we overwrite the values
            # for the requests that have their own seeds.
            q.exponential_()
            if generators:
                for i, generator in generators.items():
                    q[i].exponential_(generator=generator)
        return probs.div_(q).argmax(dim=-1).view(-1)

    def _ttnn_sampling_padded(
        self,
        filtered_logits: torch.Tensor,
        candidate_indices: torch.Tensor,
        sampling_metadata: XLASupportedSamplingMetadata,
    ) -> torch.Tensor:
        """Use fused ttnn.sampling kernel for non-greedy sampling.

        The kernel does softmax + top-k + top-p + multinomial in one fused
        call on the pre-filtered candidate set (~128 tokens), avoiding the
        scatter-back to full vocab and the compiled softmax/Gumbel-max chain.
        """
        batch = filtered_logits.shape[0]
        pad_batch = _TTNN_SAMPLING_BATCH_SIZE

        values = filtered_logits.to(torch.bfloat16)
        indices = candidate_indices.to(torch.int32)

        # tt::sampling writer kernel's valid k range is 1..nearest32_K (32).
        # k > 32 causes out-of-bounds L1 reads; use _TOPK_K_PER_CHUNK as cap.
        if sampling_metadata.top_k is not None:
            k_tensor = (
                sampling_metadata.top_k[:batch]
                .to(torch.int32)
                .clamp(max=_TOPK_K_PER_CHUNK)
            )
        else:
            k_tensor = torch.full(
                (batch,), _TOPK_K_PER_CHUNK, dtype=torch.int32, device=values.device
            )

        if sampling_metadata.top_p is not None:
            p_tensor = sampling_metadata.top_p[:batch].to(torch.bfloat16)
        else:
            p_tensor = torch.ones(batch, dtype=torch.bfloat16, device=values.device)

        # Kernel expects 1/temperature; use 1.0 for greedy requests.
        raw_temp = sampling_metadata.temperature[:batch]
        is_greedy = raw_temp < _SAMPLING_EPS
        temp_recip = torch.where(is_greedy, torch.ones_like(raw_temp), 1.0 / raw_temp)
        temp_tensor = temp_recip.to(torch.bfloat16)

        # Pad batch to 32 (kernel requirement).
        if batch < pad_batch:
            pad_size = pad_batch - batch
            values = torch.nn.functional.pad(
                values, (0, 0, 0, pad_size), value=float("-inf")
            )
            indices = torch.nn.functional.pad(indices, (0, 0, 0, pad_size))
            k_tensor = torch.nn.functional.pad(k_tensor, (0, pad_size), value=1)
            p_tensor = torch.nn.functional.pad(p_tensor, (0, pad_size), value=1.0)
            temp_tensor = torch.nn.functional.pad(temp_tensor, (0, pad_size), value=1.0)

        # seed=0: skip rand_tile_init so hardware RNG advances naturally each step.
        result = torch.ops.tt.sampling(values, indices, k_tensor, p_tensor, temp_tensor)

        return result.view(-1)


def apply_top_k_top_p_fast(
    logits: torch.Tensor,
    k: torch.Tensor | None,
    p: torch.Tensor | None,
    pad_to_batch: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Top-k/top-p filtering via multi-core ttnn.topk.

    Splits vocab into power-of-2 chunks (<= 32768) so torch.topk compiles
    to multi-core ttnn.topk (~0.18ms/chunk) instead of single-core ttnn.sort
    (~9ms). Returns (filtered_logits, candidate_indices) where both tensors
    have shape [batch, num_chunks * k_per_chunk] and candidate_indices holds
    global vocab positions.

    The top-k and top-p filters are applied on the small candidate set
    (~128 tokens) rather than the full vocab.
    """
    batch, vocab_size = logits.shape[0], logits.shape[-1]
    chunk_size, padded_chunk_size = _get_topk_split_params(vocab_size)

    # Pad batch to _TTNN_SAMPLING_BATCH_SIZE (32) so topk runs on
    # [32, chunk] instead of [batch, chunk]. Multi-core topk is 14x faster
    # at batch=32 — hardware processes all rows simultaneously and
    # dummy -inf rows add negligible overhead.
    # Only pad when k/p filtering is skipped (k=None, p=None) — if actual
    # k/p tensors are passed they have shape [batch] and won't broadcast
    # against the padded [32, ...] topk outputs.
    _pad_for_topk = (
        pad_to_batch and k is None and p is None and batch < _TTNN_SAMPLING_BATCH_SIZE
    )
    if _pad_for_topk:
        logits = torch.nn.functional.pad(
            logits,
            (0, 0, 0, _TTNN_SAMPLING_BATCH_SIZE - batch),
            value=float("-inf"),
        )

    # Split vocab, pad each chunk to power-of-2, run topk.
    chunks = torch.split(logits, chunk_size, dim=-1)
    topk_values_list = []
    topk_indices_list = []
    for i, chunk in enumerate(chunks):
        if chunk.shape[-1] < padded_chunk_size:
            chunk = torch.nn.functional.pad(
                chunk, (0, padded_chunk_size - chunk.shape[-1]), value=float("-inf")
            )
        vals, inds = torch.topk(chunk, k=_TOPK_K_PER_CHUNK, dim=-1)
        topk_values_list.append(vals)
        # Offset local chunk indices to global vocab positions.
        topk_indices_list.append(inds + i * chunk_size)

    # Concat: [batch, num_chunks * k_per_chunk]
    all_values = torch.cat(topk_values_list, dim=-1)
    all_indices = torch.cat(topk_indices_list, dim=-1)

    # Apply top-k and top-p on the small candidate set.
    if k is not None or p is not None:
        probs = all_values.softmax(dim=-1)
        probs_sort, _ = probs.sort(dim=-1, descending=False)

        if k is not None:
            top_k_count = probs_sort.size(1) - k.to(torch.long)
            top_k_count = top_k_count.clamp(max=probs_sort.size(1) - 1).unsqueeze(1)
            top_k_cutoff = probs_sort.gather(-1, top_k_count)
            no_top_k_mask = ((k <= 0) | (k >= vocab_size)).unsqueeze(1)
            top_k_cutoff.masked_fill_(no_top_k_mask, -float("inf"))
            all_values = all_values.masked_fill(probs < top_k_cutoff, -float("inf"))

        if p is not None:
            cumprob = torch.cumsum(probs_sort, dim=-1)
            top_p_mask = cumprob <= 1 - p.unsqueeze(1)
            top_p_mask[:, -1] = False
            top_p_count = top_p_mask.sum(dim=-1).unsqueeze(1)
            top_p_cutoff = probs_sort.gather(-1, top_p_count)
            all_values = all_values.masked_fill(probs < top_p_cutoff, -float("inf"))

    # Trim back to original batch if we padded above.
    if _pad_for_topk:
        return all_values[:batch], all_indices[:batch]
    return all_values, all_indices
