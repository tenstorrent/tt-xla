# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""CPU tests for TTDiffusionGemmaModelState.

Covers the config plumbing, the request lifecycle over the diffusion slot table,
and the per-request additive attention mask (encoder rows causal, denoise rows
bidirectional) that replaces upstream's per-request causal tensor.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from vllm_tt.diffusion_gemma import _MASK_BLOCK, _MASK_KEEP, TTDiffusionGemmaModelState

CANVAS = 8
HIDDEN = 4
VOCAB = 32
MAX_REQS = 4


def make_state(canvas_length=CANVAS, max_denoising_steps=None, stability=1):
    gen_config = {"stability_threshold": stability}
    if max_denoising_steps is not None:
        gen_config["max_denoising_steps"] = max_denoising_steps

    model_config = SimpleNamespace(
        max_model_len=64,
        dtype=torch.float32,
        hf_text_config=SimpleNamespace(hidden_size=HIDDEN),
        get_vocab_size=lambda: VOCAB,
        get_inputs_embeds_size=lambda: HIDDEN,
        try_get_generation_config=lambda: gen_config,
    )
    vllm_config = SimpleNamespace(
        model_config=model_config,
        scheduler_config=SimpleNamespace(
            max_num_seqs=MAX_REQS, max_num_batched_tokens=64
        ),
        diffusion_config=SimpleNamespace(
            canvas_length=canvas_length, max_denoising_steps=None
        ),
    )
    return TTDiffusionGemmaModelState(
        vllm_config, torch.nn.Identity(), None, torch.device("cpu")
    )


def req(req_id, prompt_len=3):
    return SimpleNamespace(req_id=req_id, prompt_token_ids=list(range(prompt_len)))


@pytest.mark.push
@pytest.mark.cpu
def test_init_plumbs_diffusion_config_and_generation_config():
    st = make_state(canvas_length=CANVAS, stability=1)
    ds = st.diffusion_states
    assert st.canvas_length == CANVAS
    assert ds.canvas_length == CANVAS
    assert ds.vocab_size == VOCAB
    # Transformers' stability_threshold counts the previous step only; the
    # history buffer here includes the current step, hence +1.
    assert ds.stability_threshold == 2
    assert ds.accepted_canvas_history.shape == (MAX_REQS, 2, CANVAS)


@pytest.mark.push
@pytest.mark.cpu
def test_init_defaults_max_denoising_steps_from_generation_config():
    st = make_state(max_denoising_steps=11)
    assert st.diffusion_states.max_denoising_steps == 11


@pytest.mark.push
@pytest.mark.cpu
def test_supported_generation_tasks_is_generate_only():
    assert make_state().get_supported_generation_tasks() == ("generate",)


@pytest.mark.push
@pytest.mark.cpu
def test_add_request_seeds_encoder_phase_and_prompt_len():
    st = make_state()
    st.add_request(2, req("A", prompt_len=5))
    ds = st.diffusion_states
    assert bool(ds.is_encoder_phase[2]) is True
    assert int(ds.prompt_len[2]) == 5
    # Canvas seeded with in-range random tokens.
    assert int(ds.canvas[2].min()) >= 0
    assert int(ds.canvas[2].max()) < VOCAB
    assert st._req_id_to_slot["A"] == 2


@pytest.mark.push
@pytest.mark.cpu
def test_add_request_skips_prompt_len_for_warmup():
    st = make_state()
    st.add_request(0, req("_warmup_0", prompt_len=5))
    assert int(st.diffusion_states.prompt_len[0]) == 0


@pytest.mark.push
@pytest.mark.cpu
def test_remove_request_clears_slot_and_mapping():
    st = make_state()
    st.add_request(1, req("A"))
    st.remove_request("A")
    assert bool(st.diffusion_states.is_encoder_phase[1]) is False
    assert "A" not in st._req_id_to_slot
    # Removing an unknown id is a no-op, not a KeyError.
    st.remove_request("nope")


@pytest.mark.push
@pytest.mark.cpu
def test_build_attn_mask_causal_for_encoder_open_for_denoise():
    st = make_state()
    st.add_request(0, req("A"))  # encoder phase
    st.add_request(1, req("B"))
    st.diffusion_states.is_encoder_phase[1] = False  # denoising

    s = 3
    mask = st.build_attn_mask(np.array([0, 1]), s)
    assert mask.shape == (2, 1, s, s)
    assert mask.dtype == torch.float32

    # Row 0 (encoder): strictly-upper triangle blocked, rest kept.
    expected_causal = torch.triu(torch.full((s, s), _MASK_BLOCK), diagonal=1)
    torch.testing.assert_close(mask[0, 0], expected_causal)

    # Row 1 (denoise): fully open, every position attends everywhere.
    torch.testing.assert_close(mask[1, 0], torch.full((s, s), _MASK_KEEP))


@pytest.mark.push
@pytest.mark.cpu
def test_build_attn_mask_follows_slot_order_not_slot_index():
    st = make_state()
    for slot in range(3):
        st.add_request(slot, req(f"r{slot}"))
    st.diffusion_states.is_encoder_phase[0] = False

    # Batch position 0 -> slot 2 (encoder), position 1 -> slot 0 (denoise).
    mask = st.build_attn_mask(np.array([2, 0]), 2)
    assert mask[0, 0, 0, 1].item() == _MASK_BLOCK
    assert mask[1, 0, 0, 1].item() == _MASK_KEEP


@pytest.mark.push
@pytest.mark.cpu
def test_prepare_attn_uses_mask_and_disables_is_causal():
    st = make_state()
    st.add_request(0, req("A"))
    page_table = torch.zeros(1, 2, dtype=torch.int32)
    cache_position = torch.ones(1, dtype=torch.int32)

    md = st.prepare_attn(
        ["layer0"],
        page_table,
        cache_position,
        slot_indices_np=np.array([0]),
        query_len=3,
    )
    meta = md["layer0"]
    assert meta.is_causal is False
    assert meta.attn_mask is not None
    assert meta.attn_mask.shape == (1, 1, 3, 3)


@pytest.mark.push
@pytest.mark.cpu
def test_prepare_attn_falls_back_to_causal_without_step_metadata():
    st = make_state()
    md = st.prepare_attn(
        ["layer0"],
        torch.zeros(1, 2, dtype=torch.int32),
        torch.ones(1, dtype=torch.int32),
    )
    meta = md["layer0"]
    assert meta.is_causal is True
    assert meta.attn_mask is None
