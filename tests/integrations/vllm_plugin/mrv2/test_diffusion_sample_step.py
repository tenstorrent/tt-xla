# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""CPU unit tests for the DiffusionGemma accept/renoise sample step.

Validates the pure-compute core of the TT DiffusionGemma port (state buffers +
diffusion_sample_step) and, in particular, that the two TT device edits are
behaviour-preserving:
- cummax(sorted_ent) -> sorted_ent (acceptance mask identical)
- randint -> (rand * vocab).long() (uniform, non-degenerate token ids)
"""

import numpy as np
import torch

from vllm_tt.diffusion_gemma import (
    DiffusionGemmaRequestStates,
    diffusion_sample_step,
)

DEVICE = torch.device("cpu")
CL = 8
VOCAB = 32
HIDDEN = 16
ST = 2
MAX_STEPS = 3
MAX_REQS = 4


def _make_states():
    return DiffusionGemmaRequestStates(
        max_num_reqs=MAX_REQS,
        canvas_length=CL,
        vocab_size=VOCAB,
        max_denoising_steps=MAX_STEPS,
        device=DEVICE,
        hidden_size=HIDDEN,
        stability_threshold=ST,
    )


def _run_step(states, logits, decode_slot, is_encoder_phase):
    """Run one step for a single decode request in slot `decode_slot`."""
    states.is_encoder_phase[decode_slot] = is_encoder_phase
    num_reqs = 1
    decode_slots = torch.tensor([decode_slot], dtype=torch.int64)
    decode_idx = torch.tensor([0], dtype=torch.int64)
    all_slots = torch.tensor([decode_slot], dtype=torch.int64)
    valid_canvas_len = torch.tensor([CL], dtype=torch.int64)
    sampled = torch.zeros(num_reqs, CL, dtype=torch.int32)
    num_sampled = torch.zeros(num_reqs, dtype=torch.int32)
    draft_tokens = torch.zeros(MAX_REQS, CL, dtype=torch.int64)
    embed_weight = torch.randn(VOCAB, HIDDEN)
    normalizer = torch.tensor(1.0)
    scaled = diffusion_sample_step(
        logits,
        decode_slots,
        decode_idx,
        all_slots,
        valid_canvas_len,
        states.canvas,
        states.argmax_canvas,
        states.step,
        states.is_encoder_phase,
        states.confident,
        states.self_conditioning_embeds,
        embed_weight,
        normalizer,
        states.accepted_canvas_history,
        states.accepted_canvas_history_len,
        sampled,
        num_sampled,
        draft_tokens,
        max_denoising_steps=float(MAX_STEPS),
        t_min=0.1,
        t_max=1.0,
        confidence_threshold=100.0,  # always "confident" for these small tests
        vocab_size=VOCAB,
        CL=CL,
        ST=ST,
        entropy_bound=1e9,  # accept everything -> deterministic canvas
        sc_vocab_start=0,
        sc_vocab_end=VOCAB,
        tp_size=1,
        tp_group_name="",
    )
    return sampled, num_sampled, draft_tokens, scaled


def test_random_tokens_uniform_and_in_range():
    states = _make_states()
    states.init_canvas(np.arange(MAX_REQS))
    assert states.canvas.min() >= 0
    assert states.canvas.max() < VOCAB
    # Non-degenerate: init_canvas replaces torch.randint (degenerate on TT) with
    # the float RNG, which must still spread across many distinct values.
    assert states.canvas.unique().numel() > VOCAB // 4


def test_commit_step_emits_argmax_canvas():
    torch.manual_seed(0)
    states = _make_states()
    states.argmax_canvas[1] = torch.arange(CL, dtype=torch.int64) % VOCAB
    logits = torch.randn(CL, VOCAB)
    sampled, num_sampled, _, _ = _run_step(states, logits, 1, is_encoder_phase=True)
    # Commit emits the stored argmax canvas and the full valid length.
    assert torch.equal(sampled[0].to(torch.int64), states.argmax_canvas[1])
    assert int(num_sampled[0]) == CL
    # After a commit the request denoises next, step reset.
    assert bool(states.is_encoder_phase[1]) is False
    assert int(states.step[1]) == 0


def test_denoise_step_commits_nothing_and_advances():
    torch.manual_seed(0)
    states = _make_states()
    logits = torch.randn(CL, VOCAB)
    sampled, num_sampled, draft_tokens, _ = _run_step(
        states, logits, 2, is_encoder_phase=False
    )
    # Denoise commits nothing but advances the step and publishes the canvas.
    assert int(num_sampled[0]) == 0
    assert torch.all(sampled == 0)
    assert int(states.step[2]) == 1
    assert torch.equal(draft_tokens[2], states.canvas[2])


def test_max_steps_forces_convergence_to_commit():
    torch.manual_seed(0)
    states = _make_states()
    states.step[3] = MAX_STEPS - 1  # one denoise away from the step cap
    logits = torch.randn(CL, VOCAB)
    _run_step(states, logits, 3, is_encoder_phase=False)
    # Hitting max_denoising_steps converges -> commit next step.
    assert int(states.step[3]) == MAX_STEPS
    assert bool(states.is_encoder_phase[3]) is True


def test_cummax_edit_matches_upstream_expression():
    """The TT edit (drop cummax) must give a bit-identical acceptance mask.

    Reproduces the acceptance mask both ways over the same entropy and asserts
    equality, mirroring exactly what diffusion_sample_step computes internally.
    """
    torch.manual_seed(1)
    token_entropy = torch.rand(5, CL)
    sorted_ent, sorted_idx = torch.sort(token_entropy, dim=-1)
    cumsum_ent = torch.cumsum(sorted_ent, dim=-1)

    bound = 0.5
    # Upstream: cumsum - cummax
    cummax_ent = torch.cummax(sorted_ent, dim=-1).values
    mask_upstream = (cumsum_ent - cummax_ent) <= bound
    # TT edit: cumsum - sorted_ent
    mask_edit = (cumsum_ent - sorted_ent) <= bound

    assert torch.equal(cummax_ent, sorted_ent)
    assert torch.equal(mask_upstream, mask_edit)


def test_step_is_deterministic_under_seed():
    logits = torch.randn(CL, VOCAB)
    results = []
    for _ in range(2):
        torch.manual_seed(7)
        states = _make_states()
        _run_step(states, logits.clone(), 0, is_encoder_phase=False)
        results.append(states.canvas[0].clone())
    assert torch.equal(results[0], results[1])
