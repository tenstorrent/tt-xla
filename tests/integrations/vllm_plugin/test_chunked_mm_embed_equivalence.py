# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Chunked multi-modal input equivalence (tt-xla #5824).

The load-bearing property of chunked multimodal prefill: the vision encoder runs
once on the whole item and its output is parked in ``encoder_cache``, then each
prefill chunk re-slices that cached tensor. Splitting the prompt must therefore
reconstruct **exactly** the embedding stream that a single-shot prefill builds --
the same encoder rows in the same positions.

That is pure index arithmetic in ``_gather_mm_embeddings`` (item-relative
``start_idx``/``end_idx``, the possibly-negative ``col_base``, the row-major
``mm_indices``) plus the ``index_copy`` scatter in ``_get_model_inputs``. Both
are exercised here on CPU against a synthetic encoder output, so the test is
independent of any particular vision tower -- notably of the Qwen2-VL encoder
saturation tracked in #5858, which makes on-device token comparison for that
model meaningless.

The complementary guarantees live elsewhere: chunk sizing/scheduling in
``unit_tests/test_ascend_scheduler_batching.py``, and cached-prefix attention in
``generative/test_chunked_prefill.py``.
"""

from types import SimpleNamespace

import pytest
import torch
from vllm.multimodal.inputs import MultiModalFeatureSpec, PlaceholderRange
from vllm_tt.model_runner import TTModelRunner

# Device-free, but the vLLM push matrix only runs device-marked jobs; tag so the
# single_device job collects these (they need no device and run in milliseconds).
pytestmark = [pytest.mark.push, pytest.mark.single_device]

_H = 8  # hidden size
_IMG_TOKEN_ID = 999  # placeholder token id occupying the image span
_PADDINGS = [1, 8, 16, 32, 64, 128, 256]


def _encoder_output(length: int, salt: float = 0.0) -> torch.Tensor:
    """Deterministic stand-in for a vision tower output, distinct per row."""
    base = torch.arange(length * _H, dtype=torch.float32).reshape(length, _H)
    return base / 1000.0 + salt


def _prompt_ids(total: int, offset: int, length: int) -> list[int]:
    """Text ids, with the image placeholder span at ``offset``."""
    ids = [(i % 97) + 1 for i in range(total)]
    for i in range(offset, offset + length):
        ids[i] = _IMG_TOKEN_ID
    return ids


class _FakeEmbedder:
    """Stands in for ``model.embed_input_ids``.

    Mirrors the real contract: embed the text ids and zero the positions marked
    multimodal (upstream masks those ids because they can be out of vocab). The
    mm rows are then written by the ``index_copy`` scatter under test.
    """

    def embed_input_ids(self, input_ids, is_multimodal=None):
        emb = input_ids.unsqueeze(-1).to(torch.float32) * 10.0 + torch.arange(
            _H, dtype=torch.float32
        )
        if is_multimodal is not None:
            emb = torch.where(is_multimodal.unsqueeze(-1), torch.zeros_like(emb), emb)
        return emb


def _runner(reqs: dict, max_num_reqs: int, encoder_cache: dict):
    """The only runner state the two methods under test read."""
    return SimpleNamespace(
        input_batch=SimpleNamespace(req_ids=list(reqs)),
        requests=reqs,
        num_tokens_paddings=_PADDINGS,
        max_num_reqs=max_num_reqs,
        encoder_cache=encoder_cache,
        device="cpu",
        supports_mm_inputs=True,
        model=_FakeEmbedder(),
    )


def _req_state(prompt_ids: list[int], num_computed: int, features: list):
    return SimpleNamespace(
        num_computed_tokens=num_computed,
        mm_features=features,
        prompt_token_ids=prompt_ids,
        num_tokens=len(prompt_ids),
    )


def _feature(identifier: str, offset: int, length: int, is_embed=None):
    return MultiModalFeatureSpec(
        data=None,
        modality="image",
        identifier=identifier,
        mm_position=PlaceholderRange(offset=offset, length=length, is_embed=is_embed),
    )


def _step_embeds(runner, reqs, scheduled: dict) -> torch.Tensor:
    """Run gather + scatter for one step; return [num_reqs, padded_width, H]."""
    sched_out = SimpleNamespace(num_scheduled_tokens=scheduled)
    mm_embed_inputs = TTModelRunner._gather_mm_embeddings(runner, sched_out)

    padded_width = mm_embed_inputs[1].shape[1]
    rows = []
    for rid in runner.input_batch.req_ids:
        st = reqs[rid]
        n = scheduled.get(rid, 0)
        start = st.num_computed_tokens
        row = st.prompt_token_ids[start : start + n]
        row = row + [0] * (padded_width - len(row))
        rows.append(row)
    input_ids = torch.tensor(rows, dtype=torch.int32)

    _, inputs_embeds = TTModelRunner._get_model_inputs(
        runner, input_ids, mm_embed_inputs
    )
    return inputs_embeds


def _single_shot(prompt_ids, features, encoder_cache) -> torch.Tensor:
    """Embedding stream for the whole prompt in one prefill step."""
    total = len(prompt_ids)
    reqs = {"r0": _req_state(prompt_ids, 0, features)}
    runner = _runner(reqs, max_num_reqs=1, encoder_cache=encoder_cache)
    return _step_embeds(runner, reqs, {"r0": total})[0, :total]


def _chunked(prompt_ids, features, encoder_cache, chunk) -> torch.Tensor:
    """Embedding stream for the same prompt split into ``chunk``-sized steps."""
    total = len(prompt_ids)
    pieces = []
    computed = 0
    while computed < total:
        n = min(chunk, total - computed)
        reqs = {"r0": _req_state(prompt_ids, computed, features)}
        runner = _runner(reqs, max_num_reqs=1, encoder_cache=encoder_cache)
        out = _step_embeds(runner, reqs, {"r0": n})
        pieces.append(out[0, :n])
        computed += n
    return torch.cat(pieces, dim=0)


# (total, offset, img_len, chunk)
_CASES = [
    # Image straddles one chunk boundary.
    (64, 8, 32, 32),
    # Image spans several whole chunks plus partial head and tail -- the case
    # that produced slices 497/512/15 on device at chunk=512.
    (128, 5, 100, 32),
    # Image starts exactly on a chunk boundary.
    (96, 32, 48, 32),
    # Image ends exactly on a chunk boundary.
    (96, 16, 48, 32),
    # Chunk smaller than the image by a lot: many interior chunks are all-image.
    (128, 3, 120, 8),
    # Image is the entire prompt.
    (64, 0, 64, 16),
    # Final chunk is a short remainder.
    (70, 9, 40, 32),
]


@pytest.mark.parametrize("total,offset,img_len,chunk", _CASES)
def test_chunked_mm_embeddings_match_single_shot(total, offset, img_len, chunk):
    """Splitting a prompt must reproduce the single-shot embedding stream exactly.

    A wrong ``start_idx``/``end_idx``/``col_base`` puts the right number of
    encoder rows in the wrong columns, or the wrong rows in the right columns --
    either way this comparison fails while token counts still look consistent.
    """
    prompt_ids = _prompt_ids(total, offset, img_len)
    features = [_feature("img0", offset, img_len)]
    cache = {"img0": _encoder_output(img_len)}

    single = _single_shot(prompt_ids, features, cache)
    multi = _chunked(prompt_ids, features, cache, chunk)

    assert single.shape == multi.shape == (total, _H)
    assert torch.equal(single, multi), (
        "chunked prefill produced a different embedding stream than single-shot "
        f"(total={total}, image=[{offset},{offset + img_len}), chunk={chunk}); "
        f"first mismatching row = {int((single != multi).any(dim=1).nonzero()[0])}"
    )


@pytest.mark.parametrize("chunk", [16, 32])
def test_chunked_mm_places_the_correct_encoder_rows(chunk):
    """Pin the mapping itself, not just self-consistency.

    Every image position must carry its own encoder row, and every text position
    must carry a text embedding. Guards against an off-by-one that shifts the
    whole image by a row (which the equivalence test alone could miss if both
    paths shifted identically).
    """
    total, offset, img_len = 96, 7, 50
    prompt_ids = _prompt_ids(total, offset, img_len)
    features = [_feature("img0", offset, img_len)]
    enc = _encoder_output(img_len)
    cache = {"img0": enc}

    multi = _chunked(prompt_ids, features, cache, chunk)

    for i in range(img_len):
        assert torch.equal(
            multi[offset + i], enc[i]
        ), f"prompt position {offset + i} should hold encoder row {i}"
    text_embed = _FakeEmbedder().embed_input_ids(
        torch.tensor(prompt_ids, dtype=torch.int32)
    )
    for pos in list(range(offset)) + list(range(offset + img_len, total)):
        assert torch.equal(
            multi[pos], text_embed[pos]
        ), f"text position {pos} should hold a text embedding, not an encoder row"


@pytest.mark.parametrize("chunk", [16, 32])
def test_chunked_mm_interleaved_placeholders_match_single_shot(chunk):
    """Same guarantee for an ``is_embed`` mask (Pixtral-style break tokens).

    Here the placeholder span interleaves real text tokens with embedding slots,
    so the token range must be mapped through ``get_embeds_indices_in_range``
    before slicing the encoder output. Splitting must not shift that mapping.
    """
    total, offset, span = 96, 5, 60
    prompt_ids = _prompt_ids(total, offset, span)
    # Every 6th position in the span is a real token (e.g. [IMG_BREAK]).
    is_embed = torch.ones(span, dtype=torch.bool)
    is_embed[5::6] = False
    num_embeds = int(is_embed.sum())
    features = [_feature("img0", offset, span, is_embed=is_embed)]
    cache = {"img0": _encoder_output(num_embeds, salt=7.0)}

    single = _single_shot(prompt_ids, features, cache)
    multi = _chunked(prompt_ids, features, cache, chunk)

    assert torch.equal(
        single, multi
    ), f"interleaved-placeholder chunking diverged from single-shot (chunk={chunk})"
    # And the embedding rows must land only on is_embed positions, in order.
    enc = cache["img0"]
    k = 0
    for i in range(span):
        if bool(is_embed[i]):
            assert torch.equal(
                multi[offset + i], enc[k]
            ), f"span position {i} should hold encoder row {k}"
            k += 1
    assert k == num_embeds


def test_chunked_mm_two_requests_match_single_shot():
    """Two same-stage requests, each with its own image.

    ``_gather_mm_embeddings`` builds a ``[max_num_reqs, padded_width]`` mask and
    flattens it row-major to index a reshaped ``[num_reqs * padded_width, H]``
    buffer, so the row count and stride must agree across both requests. With one
    request that is trivially true; this is the case that would catch a stride or
    concatenation-order bug.
    """
    total, chunk = 96, 32
    specs = {
        "r0": (7, 50),  # (offset, img_len)
        "r1": (20, 40),
    }
    prompts = {r: _prompt_ids(total, o, n) for r, (o, n) in specs.items()}
    features = {r: [_feature(f"{r}-img", o, n)] for r, (o, n) in specs.items()}
    cache = {
        f"{r}-img": _encoder_output(n, salt=float(i))
        for i, (r, (o, n)) in enumerate(specs.items())
    }

    def run(chunk_size):
        pieces = {r: [] for r in specs}
        computed = 0
        while computed < total:
            n = min(chunk_size, total - computed)
            reqs = {r: _req_state(prompts[r], computed, features[r]) for r in specs}
            runner = _runner(reqs, max_num_reqs=len(specs), encoder_cache=cache)
            out = _step_embeds(runner, reqs, {r: n for r in specs})
            for idx, r in enumerate(runner.input_batch.req_ids):
                pieces[r].append(out[idx, :n])
            computed += n
        return {r: torch.cat(v, dim=0) for r, v in pieces.items()}

    single = run(total)
    multi = run(chunk)

    for r in specs:
        assert torch.equal(
            single[r], multi[r]
        ), f"request {r} diverged between single-shot and chunk={chunk}"
    # The two requests must not have swapped embeddings.
    o0, n0 = specs["r0"]
    assert torch.equal(
        multi["r0"][o0 : o0 + n0], cache["r0-img"]
    ), "r0 did not receive its own encoder output"
