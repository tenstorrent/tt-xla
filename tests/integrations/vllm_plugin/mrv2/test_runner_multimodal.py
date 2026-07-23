# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""CPU unit tests for MRv2 multimodal host logic.

``TTModelRunnerV2._gather_mm_embeddings`` builds the per-pass mm-embed mask and
flat scatter indices, and ``TTModelState.get_mm_embeddings`` merges the gathered
encoder embeddings into the text embeddings with a static ``index_copy`` (not
``masked_scatter_``). These tests pin the mask layout / index math and the merge
purely on CPU with a fake encoder cache and a fake embed_input_ids.
"""
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from vllm_tt.model_runner_v2 import TTModelRunnerV2
from vllm_tt.model_state import TTModelState

HID = 4


def pos_info(offset, length):
    # Whole-item placeholder (is_embed None); identity embed index mapping.
    return SimpleNamespace(
        offset=offset,
        length=length,
        is_embed=None,
        get_embeds_indices_in_range=lambda s, e: (s, e),
    )


def mm_feature(identifier, offset, length):
    return SimpleNamespace(identifier=identifier, mm_position=pos_info(offset, length))


def gather_runner(mm_features_by_slot, num_computed_by_slot, encoder_cache):
    r = object.__new__(TTModelRunnerV2)
    r.device = torch.device("cpu")
    r.mm_features_by_slot = mm_features_by_slot
    r.encoder_cache = encoder_cache
    n = max(num_computed_by_slot) + 1 if num_computed_by_slot else 1
    r.req_states = SimpleNamespace(num_computed_tokens=np.zeros(8, dtype=np.int32))
    for slot, c in num_computed_by_slot.items():
        r.req_states.num_computed_tokens[slot] = c
    return r


@pytest.mark.push
@pytest.mark.cpu
def test_gather_mm_embeddings_mask_and_indices():
    # Request at slot 2, image placeholder at prompt positions [0, 3) (3 tokens),
    # scheduled as a fresh prefill of 4 tokens at batch position 0.
    enc = torch.arange(3 * HID, dtype=torch.float32).reshape(3, HID)
    r = gather_runner(
        mm_features_by_slot={2: [mm_feature("h0", offset=0, length=3)]},
        num_computed_by_slot={2: 0},
        encoder_cache={"h0": enc},
    )
    idx = np.array([2], dtype=np.int32)
    nst = np.array([4], dtype=np.int32)

    mm_embeds, is_mm_embed, mm_indices = r._gather_mm_embeddings(idx, nst, 4)

    # Mask: row 0, columns 0..2 are mm tokens; column 3 is text.
    assert is_mm_embed.shape == (1, 4)
    assert is_mm_embed[0].tolist() == [True, True, True, False]
    # Flat row-major positions of the True cells (row0 cols 0,1,2).
    assert mm_indices.tolist() == [0, 1, 2]
    # One gathered embedding block of the 3 image tokens.
    assert len(mm_embeds) == 1
    assert torch.equal(mm_embeds[0], enc)


@pytest.mark.push
@pytest.mark.cpu
def test_gather_mm_embeddings_no_features_is_empty():
    r = gather_runner({}, {0: 0}, {})
    idx = np.array([0], dtype=np.int32)
    nst = np.array([2], dtype=np.int32)
    mm_embeds, is_mm_embed, mm_indices = r._gather_mm_embeddings(idx, nst, 2)
    assert mm_embeds == []
    assert not is_mm_embed.any()
    assert mm_indices.numel() == 0


@pytest.mark.push
@pytest.mark.cpu
def test_get_mm_embeddings_index_copy_merge():
    # Fake model: embed_input_ids returns token id broadcast to HID; the merge
    # replaces the masked rows with the encoder embeddings via index_copy.
    ms = object.__new__(TTModelState)

    def embed_input_ids(input_ids, is_multimodal=None):
        # [reqs, tokens] -> [reqs, tokens, HID], value = token id.
        return input_ids.unsqueeze(-1).to(torch.float32).expand(-1, -1, HID).clone()

    ms.model = SimpleNamespace(embed_input_ids=embed_input_ids)

    input_ids = torch.tensor([[5, 6, 7, 8]], dtype=torch.int32)  # [1, 4]
    is_mm_embed = torch.tensor([[True, True, True, False]])
    mm_indices = torch.tensor([0, 1, 2], dtype=torch.int64)
    mm_flat = torch.arange(3 * HID, dtype=torch.float32).reshape(3, HID)

    out = ms.get_mm_embeddings(input_ids, [mm_flat], is_mm_embed, mm_indices)

    assert out.shape == (1, 4, HID)
    # First three rows replaced by the encoder embeddings.
    assert torch.equal(out.reshape(-1, HID)[:3], mm_flat)
    # Last (text) row is the token-id embedding for token 8.
    assert torch.equal(out[0, 3], torch.full((HID,), 8.0))


@pytest.mark.push
@pytest.mark.cpu
def test_get_mm_embeddings_text_only_no_merge():
    ms = object.__new__(TTModelState)
    ms.model = SimpleNamespace(
        embed_input_ids=lambda ids, is_multimodal=None: torch.zeros((*ids.shape, HID))
    )
    input_ids = torch.tensor([[1, 2]], dtype=torch.int32)
    is_mm_embed = torch.zeros((1, 2), dtype=torch.bool)
    # Empty mm_embeds -> pure text embeddings, no index_copy.
    out = ms.get_mm_embeddings(
        input_ids, [], is_mm_embed, torch.tensor([], dtype=torch.int64)
    )
    assert out.shape == (1, 2, HID)
    assert torch.count_nonzero(out) == 0
