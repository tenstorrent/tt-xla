# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sampler layer implementing XLA supported operations."""

import math

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
        *,
        vocab_sharded: bool = False,
    ) -> SamplerOutput:
        # Use float32 for the logits.
        logits = logits.to(torch.float32)
        # Sample the next token.
        sampled = self.sample(logits, sampling_metadata, vocab_sharded=vocab_sharded)

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

    def greedy_sample(
        self, logits: torch.Tensor, *, vocab_sharded: bool = False
    ) -> torch.Tensor:
        if vocab_sharded:
            from tt_torch.composite_ops import composite_topk_indices

            # argmax over a vocab-sharded tensor would return shard-local
            # indices. composite_topk_indices(k=1) is sharding-aware via the
            # tenstorrent.topk_indices custom sharding rule — local topk +
            # all-gather + shard-offset + merge — and returns the global
            # argmax. We use the indices-only variant because the two-output
            # `composite_topk` with a discarded `values` output crashes
            # torch_xla's BuildStableHLOCompositePass.
            indices = composite_topk_indices(
                logits, k=1, dim=-1, largest=True, sorted=False
            )
            return indices.squeeze(-1)
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
        *,
        vocab_sharded: bool = False,
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

        # Skip greedy_sample when every row is sampling (all_random=True);
        # the torch.where below would discard greedy_sampled anyway. ArgMax
        # over full vocab was ~34% of sampler runtime at b=32 in tracy.
        all_random = sampling_metadata.all_random
        if not all_random:
            greedy_sampled = self.greedy_sample(logits, vocab_sharded=vocab_sharded)

        # Build the candidate set via chunked multi-core topk. The fused
        # tt::sampling kernel applies user top-k, top-p, softmax, and
        # multinomial downstream — no need to filter twice here.
        filtered_logits, candidate_indices = chunked_topk_candidates(
            logits, vocab_sharded=vocab_sharded
        )
        random_sampled = self._ttnn_sampling_padded(
            filtered_logits, candidate_indices, sampling_metadata
        )
        if all_random:
            return random_sampled
        return torch.where(
            sampling_metadata.temperature < _SAMPLING_EPS,
            greedy_sampled,
            random_sampled,
        )

    def compute_logprobs(self, logits: torch.Tensor) -> torch.Tensor:
        return logits.log_softmax(dim=-1, dtype=torch.float32)

    def gather_logprobs(
        self,
        logprobs: torch.Tensor,
        num_logprobs: int,
        token_ids: torch.Tensor,
        *,
        replicate_anchor_mesh=None,
    ) -> LogprobsTensors:
        """
        Gather logprobs for topk and sampled/prompt token.

        Args:
          logprobs: (num tokens) x (vocab) tensor
          num_logprobs: minimum number of logprobs to
                        retain per token
          token_ids: prompt tokens (if prompt logprobs)
                     or sampled tokens (if sampled
                     logprobs); either a 1D tensor of
                     (num tokens) elements or a 2D
                     [num tokens, 1] tensor. The 2D form
                     lets compiled callers skip a
                     squeeze/unsqueeze pair around this
                     call — important on 1D-mesh TP where
                     that reshape triggers an unlowered
                     collective_permute (tt-mlir#3370).
          replicate_anchor_mesh: when set, anchor the shape-changed
                     intermediates (unsqueeze, topk, gather, cat) to
                     replicated on this mesh. Callers pass the mesh on
                     1D-mesh TP setups where Shardy otherwise re-infers
                     "model"-axis shardings on these fresh tensors and
                     emits a collective_permute that tt-mlir cannot
                     lower (tt-mlir#3370).

        Returns:
          Top-k int indices tensor, (num tokens) x (num_logprobs + 1)
          Top-k float logprobs tensor, (num tokens) x (num_logprobs + 1)
          Sampled token rank tensor, (num tokens)
        """
        if replicate_anchor_mesh is not None:
            from tt_torch.sharding import sharding_constraint_tensor

            def _anchor(t, spec):
                return sharding_constraint_tensor(t, replicate_anchor_mesh, spec)

        else:

            def _anchor(t, spec):
                return t

        # Find the topK values.
        topk_logprobs, topk_indices = torch.topk(logprobs, num_logprobs, dim=-1)
        topk_logprobs = _anchor(topk_logprobs, (None, None))
        topk_indices = _anchor(topk_indices, (None, None))

        # Get the logprob of the prompt or sampled token. Accept 2D
        # [num_tokens, 1] from compiled callers as-is to avoid a
        # squeeze/unsqueeze reshape that Shardy materializes as
        # collective_permute on 1D-mesh TP.
        if token_ids.ndim == 1:
            token_ids = token_ids.unsqueeze(-1)
        token_ids = _anchor(token_ids, (None, None))
        token_logprobs = logprobs.gather(-1, token_ids)
        token_logprobs = _anchor(token_logprobs, (None, None))

        token_ranks = count_tokens_ge(logprobs, token_logprobs)
        token_ranks = _anchor(token_ranks, (None,))

        # Cast to int32 for LogprobsTensors (vLLM convention).
        indices = torch.cat(
            (token_ids.to(torch.int32), topk_indices.to(torch.int32)),
            dim=1,
        )
        indices = _anchor(indices, (None, None))
        token_ranks = token_ranks.to(torch.int32)
        logprobs = torch.cat((token_logprobs, topk_logprobs), dim=1)
        logprobs = _anchor(logprobs, (None, None))

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

        values = filtered_logits.to(torch.bfloat16)
        indices = candidate_indices.to(torch.int32)

        # Kernel writer's valid k range is 1..nearest32_K (32); k > 32
        # causes out-of-bounds L1 reads, so cap at _TOPK_K_PER_CHUNK.
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

        # Kernel expects 1/temperature; use 1.0 for greedy rows (where the
        # outer torch.where will discard the random result anyway).
        raw_temp = sampling_metadata.temperature[:batch]
        is_greedy = raw_temp < _SAMPLING_EPS
        temp_tensor = torch.where(
            is_greedy, torch.ones_like(raw_temp), 1.0 / raw_temp
        ).to(torch.bfloat16)

        # Pad batch to 32 (kernel requirement).
        if batch < _TTNN_SAMPLING_BATCH_SIZE:
            pad_size = _TTNN_SAMPLING_BATCH_SIZE - batch
            values = torch.nn.functional.pad(
                values, (0, 0, 0, pad_size), value=float("-inf")
            )
            indices = torch.nn.functional.pad(indices, (0, 0, 0, pad_size))
            k_tensor = torch.nn.functional.pad(k_tensor, (0, pad_size), value=1)
            p_tensor = torch.nn.functional.pad(p_tensor, (0, pad_size), value=1.0)
            temp_tensor = torch.nn.functional.pad(temp_tensor, (0, pad_size), value=1.0)

        result = torch.ops.tt.sampling(values, indices, k_tensor, p_tensor, temp_tensor)
        return result[:batch].to(torch.int64)


_SHARDED_TOPK_CANDIDATES = 128  # matches typical num_chunks * k_per_chunk


def chunked_topk_candidates(
    logits: torch.Tensor,
    *,
    vocab_sharded: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a per-row top-K candidate set.

    Replicated path: splits vocab into power-of-2 chunks (<= 32768) so
    torch.topk compiles to multi-core ttnn.topk (~0.18ms/chunk) instead of
    single-core ttnn.sort (~9ms). Returns ~128 candidates total.

    Vocab-sharded path (2D-mesh TP): uses composite_topk so the tt-mlir
    custom sharding rule emits local topk + all-gather + shard-offset + merge
    instead of all-gathering the full vocab. Only the merged [batch, 128]
    candidate set crosses the wire.

    Returns (candidate_values, candidate_indices) of shape [batch, 128];
    candidate_indices holds global vocab positions. User-specified top-k /
    top-p / softmax / multinomial are applied downstream by the fused
    tt::sampling kernel, not here.
    """
    batch = logits.shape[0]

    # Multi-core topk is 14x faster at batch=32 vs small batches — pad
    # with -inf rows so dummy entries can't win the topk.
    logits = torch.nn.functional.pad(
        logits,
        (0, 0, 0, _TTNN_SAMPLING_BATCH_SIZE - batch),
        value=float("-inf"),
    )

    if vocab_sharded:
        from tt_torch.composite_ops import composite_topk

        # Vocab is sharded along the "model" axis. The composite carries the
        # sharding through tt-mlir's custom rule (local topk + all-gather of
        # candidates + shard-offset + merge). Output W = 128 is already
        # power-of-2 with W/32 = 4 >= 2, satisfying the tt::sampling kernel
        # constraints (#4560).
        all_values, all_indices = composite_topk(
            logits, k=_SHARDED_TOPK_CANDIDATES, dim=-1, largest=True, sorted=False
        )
        return all_values[:batch], all_indices[:batch]

    chunk_size, padded_chunk_size = _get_topk_split_params(logits.shape[-1])

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

    # Pad W so that Wt = W/32 is a power of 2 AND Wt >= 2 to avoid the
    # tt::sampling kernel hang (#4560). -inf values can't win the topk /
    # multinomial draw, so output is unchanged.
    cur_w = all_values.shape[-1]
    target_w = max(64, _next_power_of_2(cur_w))
    if cur_w < target_w:
        pad = target_w - cur_w
        all_values = torch.nn.functional.pad(all_values, (0, pad), value=float("-inf"))
        all_indices = torch.nn.functional.pad(all_indices, (0, pad), value=0)

    return all_values[:batch], all_indices[:batch]
