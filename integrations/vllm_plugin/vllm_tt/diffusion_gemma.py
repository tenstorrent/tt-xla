# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""DiffusionGemma decode state + accept/renoise sampler math for TT MRv2.

DiffusionGemma is a discrete diffusion LM on a Gemma-4 MoE backbone. It denoises
a fixed ``canvas_length`` token canvas over up to ``max_denoising_steps``
iterations rather than generating autoregressively. One engine step runs one
forward over all canvas positions and advances the per-request diffusion state
machine (prefill -> denoise* -> commit), reusing the spec-decode data path with
overloaded semantics.

This module is a port of upstream ``vllm.model_executor.models.diffusion_gemma``
(the CUDA/GPU runner reference) onto the Tenstorrent MRv2 data path. This first
piece is the pure-compute core -- the per-request state buffers and the
accept/renoise/convergence step -- which is framework-agnostic torch and is
validated on CPU. The runner-coupled pieces (DiffusionGemmaModelState,
DiffusionSampler, per-request attention mask, read-only KV) build on top.

Two device edits vs upstream, both behaviour-preserving (verified on TT + CPU):
- ``torch.cummax`` fails to lower on TT. It runs over ascending-sorted entropy,
  where cummax is the identity, so ``cumsum - cummax`` == ``cumsum - sorted``.
- ``torch.randint`` is degenerate on TT (few distinct values); the float RNG is
  fine, so random tokens come from ``(rand * vocab).long()``.
"""

from __future__ import annotations

import numpy as np
import torch


def _random_tokens(shape, vocab_size: int, device, dtype) -> torch.Tensor:
    """Uniform token ids in [0, vocab_size).

    Uses the float RNG rather than ``torch.randint`` (degenerate on TT).
    """
    return (torch.rand(*shape, device=device) * vocab_size).to(dtype)


def _compute_num_rejected(
    num_logits: torch.Tensor,
    num_sampled: torch.Tensor,
    query_start_loc: torch.Tensor,
) -> torch.Tensor:
    """Rejected-token count per request for the spec-decode accounting.

    A denoise step commits nothing (num_sampled == 0 while num_logits > 0), so
    its whole query span is "rejected"; a commit step rejects num_logits minus
    the committed count.
    """
    query_lens = query_start_loc[1:] - query_start_loc[:-1]
    num_rejected = num_logits - num_sampled
    is_denoise = (num_logits > 0) & (num_sampled == 0)
    return torch.where(is_denoise, query_lens, num_rejected)


def diffusion_sample_step(
    # Logits from the model [num_decode * CL, vocab]
    logits: torch.Tensor,
    # Request mapping
    decode_slots: torch.Tensor,  # [num_decode] int64 -> slot indices
    decode_idx: torch.Tensor,  # [num_decode] int64 -> position in num_reqs
    all_slots: torch.Tensor,  # [num_reqs] int64 -> all slot indices
    valid_canvas_len: torch.Tensor,  # [num_decode] int64 -> real canvas len (<=CL)
    # State tensors (modified in-place)
    canvas: torch.Tensor,  # [max_num_reqs, CL]
    argmax_canvas: torch.Tensor,  # [max_num_reqs, CL]
    step_tensor: torch.Tensor,  # [max_num_reqs]
    is_encoder_phase: torch.Tensor,  # [max_num_reqs]
    confident_tensor: torch.Tensor,  # [max_num_reqs]
    sc_embeds: torch.Tensor,  # [max_num_reqs, CL, hidden]
    embed_weight: torch.Tensor,  # [vocab, hidden]
    normalizer: torch.Tensor,
    history: torch.Tensor,  # [max_num_reqs, ST, CL]
    history_len_tensor: torch.Tensor,  # [max_num_reqs]
    # Output tensors (modified in-place)
    sampled: torch.Tensor,  # [num_reqs, CL]
    num_sampled: torch.Tensor,  # [num_reqs]
    draft_tokens: torch.Tensor,  # [max_num_reqs, >=CL]
    # Scalar config
    max_denoising_steps: float,
    t_min: float,
    t_max: float,
    confidence_threshold: float,
    vocab_size: int,
    CL: int,
    ST: int,
    entropy_bound: float,
    # Tensor-parallel vocab sharding for the self-conditioning matmul.
    sc_vocab_start: int,
    sc_vocab_end: int,
    tp_size: int,
    tp_group_name: str,
) -> torch.Tensor:
    """One denoise/commit step: temperature -> Gumbel sample -> confidence ->
    accept/renoise -> convergence, all vectorised.

    Returns the temperature-scaled logits ``[num_decode, CL, vocab]`` so the
    caller can compute logprobs outside any compiled region.
    """
    num_decode = decode_slots.shape[0]
    device = decode_slots.device

    # ---- Phase 1: Temperature schedule ----
    steps_f = step_tensor[decode_slots].float()
    remaining = (max_denoising_steps - steps_f).clamp(min=1.0)
    temp = t_min + (t_max - t_min) * (remaining / max_denoising_steps)

    # ---- Phase 2: Temperature scaling + Gumbel-max sampling ----
    logits_3d = logits.reshape(num_decode, CL, -1).float()
    scaled = logits_3d / temp[:, None, None].clamp(min=1e-10)

    # argmax(logits/T + Gumbel) ~ sample from softmax(logits/T); zero noise at T==0.
    u = torch.rand_like(scaled).clamp(min=1e-20)
    gumbel = -torch.log(-torch.log(u))
    noisy = scaled + gumbel * (temp[:, None, None] > 0).float()
    new_tokens = noisy.view(-1, noisy.shape[-1]).argmax(dim=-1).view(num_decode, CL)
    argmax_tokens = (
        scaled.view(-1, scaled.shape[-1]).argmax(dim=-1).view(num_decode, CL)
    )

    # ---- Phase 3: Probs, confidence ----
    log_probs = scaled.log_softmax(dim=-1)
    probs = log_probs.exp()
    token_entropy = -(probs * log_probs).sum(dim=-1)  # [num_decode, CL]
    # Padded rows (canvas truncated near max_model_len) are uniform: max entropy
    # (no premature convergence) and argmax 0; only valid_canvas_len is committed.
    mean_entropy = token_entropy.mean(dim=-1)
    confident_tensor[decode_slots] = mean_entropy < confidence_threshold

    # ---- Phase 4: Entropy-bound acceptance mask ----
    # cumsum - cummax over ascending-sorted entropy; cummax of an ascending
    # sequence is the sequence itself, so use sorted_ent directly (TT: cummax
    # does not lower).
    sorted_ent, sorted_idx = torch.sort(token_entropy, dim=-1)
    cumsum_ent = torch.cumsum(sorted_ent, dim=-1)
    sorted_mask = (cumsum_ent - sorted_ent) <= entropy_bound
    eb_mask = torch.zeros_like(sorted_mask)
    eb_mask.scatter_(1, sorted_idx, sorted_mask)

    # ---- Phase 5: Post-sample ----
    is_commit = is_encoder_phase[decode_slots]  # [num_decode]
    is_denoise = ~is_commit
    cur_step = step_tensor[decode_slots].float()

    # +1 for denoise, reset to 0 for commit.
    new_step_val = torch.where(
        is_denoise,
        (cur_step + 1).to(step_tensor.dtype),
        step_tensor.new_zeros(num_decode),
    )
    step_tensor[decode_slots] = new_step_val

    random_tokens = _random_tokens((num_decode, CL), vocab_size, device, canvas.dtype)
    denoise_canvas = torch.where(eb_mask, new_tokens, random_tokens)

    # Canvas: commit -> random reinit, denoise -> accept/renoise result.
    canvas[decode_slots] = torch.where(
        is_commit.unsqueeze(1), random_tokens, denoise_canvas
    )

    # History: write argmax for denoise requests at the circular slot.
    hist_len = history_len_tensor[decode_slots]
    write_pos = hist_len % ST
    for i in range(ST):
        write_here = ((write_pos == i) & is_denoise).unsqueeze(1)
        history[decode_slots, i] = torch.where(
            write_here, argmax_tokens, history[decode_slots, i]
        )

    argmax_canvas[decode_slots] = torch.where(
        is_denoise.unsqueeze(1), argmax_tokens, argmax_canvas[decode_slots]
    )

    new_hist_len = torch.where(is_denoise, hist_len + 1, hist_len.new_zeros(num_decode))
    history_len_tensor[decode_slots] = new_hist_len

    # Commit -> emit argmax_canvas; denoise -> 0. Commit only the real length.
    sampled[decode_idx] = argmax_canvas[decode_slots].to(
        sampled.dtype
    ) * is_commit.unsqueeze(1).to(sampled.dtype)
    num_sampled[decode_idx] = is_commit.to(num_sampled.dtype) * valid_canvas_len.to(
        num_sampled.dtype
    )

    # ---- Phase 6: Stability + convergence ----
    ref = history[decode_slots, 0]
    mismatch = torch.zeros(num_decode, device=device, dtype=torch.int32)
    for h in range(1, ST):
        mismatch = mismatch + (ref != history[decode_slots, h]).sum(dim=-1).int()
    stable = mismatch == 0

    step_after = step_tensor[decode_slots]
    converged = (stable & confident_tensor[decode_slots] & (new_hist_len >= ST)) | (
        step_after >= max_denoising_steps
    )
    # commit -> denoise next (False); denoise converged -> commit next (True).
    is_encoder_phase[decode_slots] = torch.where(
        is_commit, is_commit.new_zeros(num_decode), converged
    )

    # SC soft embed (probs @ embed_weight * normalizer) for slots denoising next.
    sc_keep = (is_denoise & ~is_encoder_phase[decode_slots])[:, None, None]
    local_probs = probs[..., sc_vocab_start:sc_vocab_end].to(embed_weight.dtype)
    soft_embeds = torch.matmul(
        local_probs, embed_weight[: sc_vocab_end - sc_vocab_start]
    )
    if tp_size > 1:
        soft_embeds = torch.ops.vllm.all_reduce(soft_embeds, group_name=tp_group_name)
    soft_embeds = soft_embeds * normalizer
    sc_embeds[decode_slots] = soft_embeds * sc_keep

    # Overwrite canvas with argmax for newly converged denoise requests.
    newly_converged = (converged & is_denoise).unsqueeze(1)
    canvas[decode_slots] = torch.where(
        newly_converged, argmax_canvas[decode_slots], canvas[decode_slots]
    )

    # ---- Phase 7: Publish canvas into the (spec-decode) draft-token buffer ----
    draft_tokens[all_slots, :CL] = canvas[all_slots]

    return scaled


class DiffusionGemmaRequestStates:
    """Pre-allocated per-request diffusion state, indexed by slot.

    Mirrors the indexed-slot pattern of MRv2 ``RequestState``.
    """

    def __init__(
        self,
        max_num_reqs: int,
        canvas_length: int,
        vocab_size: int,
        max_denoising_steps: int,
        device: torch.device,
        hidden_size: int,
        stability_threshold: int,
    ):
        self.max_num_reqs = max_num_reqs
        self.canvas_length = canvas_length
        self.vocab_size = vocab_size
        self.max_denoising_steps = max_denoising_steps
        self.stability_threshold = stability_threshold
        self.device = device

        self.is_encoder_phase = torch.zeros(
            max_num_reqs, dtype=torch.bool, device=device
        )
        self.canvas = torch.zeros(
            max_num_reqs, canvas_length, dtype=torch.int64, device=device
        )
        self.step = torch.zeros(max_num_reqs, dtype=torch.int32, device=device)
        self.accepted_canvas_history = torch.zeros(
            max_num_reqs,
            stability_threshold,
            canvas_length,
            dtype=torch.int64,
            device=device,
        )
        self.accepted_canvas_history_len = torch.zeros(
            max_num_reqs, dtype=torch.int32, device=device
        )
        # Latest argmax(processed_logits) per slot -- what we COMMIT (kept
        # separate from canvas, which is renoised in place during denoise).
        self.argmax_canvas = torch.zeros(
            max_num_reqs, canvas_length, dtype=torch.int64, device=device
        )
        self.prompt_len = torch.zeros(max_num_reqs, dtype=torch.int32, device=device)
        self.confident = torch.zeros(max_num_reqs, dtype=torch.bool, device=device)
        # probs @ embed_weight from the prior denoise step (masked to 0 by the
        # sampler for slots not denoising this step). Storing the [.., hidden]
        # soft embed rather than the full [.., vocab] probs shrinks the buffer.
        self.self_conditioning_embeds = torch.zeros(
            max_num_reqs, canvas_length, hidden_size, dtype=torch.float32, device=device
        )

    def init_canvas(self, slot_indices_np: np.ndarray) -> None:
        """Seed the given slots' canvas with uniform random tokens."""
        n = int(slot_indices_np.shape[0])
        self.canvas[slot_indices_np] = _random_tokens(
            (n, self.canvas_length), self.vocab_size, self.device, torch.int64
        )

    def add_request(self, slot_idx: int) -> None:
        self.is_encoder_phase[slot_idx] = True
        self.init_canvas(np.array([slot_idx]))
        self.step[slot_idx] = 0
        self.accepted_canvas_history_len[slot_idx] = 0
        self.self_conditioning_embeds[slot_idx] = 0

    def remove_request(self, slot_idx: int) -> None:
        self.is_encoder_phase[slot_idx] = False
        self.accepted_canvas_history_len[slot_idx] = 0
        self.self_conditioning_embeds[slot_idx] = 0
