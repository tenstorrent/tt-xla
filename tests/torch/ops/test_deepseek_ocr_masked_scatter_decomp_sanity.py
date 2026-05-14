# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""CPU sanity: DeepSeek-OCR image merge ``masked_scatter`` vs ``decompositions.masked_scatter``.

Synthetic shapes (push) plus optional real ``ModelLoader`` inputs (nightly): captures
``(inputs_embeds[row], mask, source)`` during the same forward as forge, then checks
PCC 1.0 vs native PyTorch and the previous inlined row-wise pattern.
"""

from __future__ import annotations

import pytest
import torch

from benchmark.utils import compute_pcc
from tt_torch.backend.decompositions import masked_scatter as masked_scatter_decomp
from utils import Category


def _deepseek_style_mask_and_source(
    seq_len: int,
    hidden: int,
    *,
    dtype: torch.dtype,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build (data [S,H], mask_1d [S], source [n_true, H]) like DeepseekOCRModel."""
    data = torch.randn(seq_len, hidden, dtype=dtype, generator=generator)
    mask_1d = torch.zeros(seq_len, dtype=torch.bool)
    # Random row-constant pattern: each row is image or not (matches images_seq_mask).
    mask_1d[torch.randperm(seq_len, generator=generator)[: seq_len // 3]] = True
    n_true = int(mask_1d.sum().item())
    if n_true == 0:
        mask_1d[0] = True
        n_true = 1
    source = torch.randn(n_true, hidden, dtype=dtype, generator=generator)
    return data, mask_1d, source


def _merge_native_main(data: torch.Tensor, mask_1d: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
    """Same as tt-forge-models DeepseekOCRModel: ``masked_scatter`` on broadcast mask."""
    return data.clone().masked_scatter(mask_1d.unsqueeze(-1), source)


def _merge_decomposition(data: torch.Tensor, mask_1d: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
    return masked_scatter_decomp(
        data.clone(),
        mask_1d.unsqueeze(-1),
        source.clone(),
    )


def _merge_inlined_rowwise_prev_fork(
    inputs_embeds_row: torch.Tensor, mask_1d: torch.Tensor, source: torch.Tensor
) -> torch.Tensor:
    """Previous inlined row-wise pattern (removed from forge in favor of ``masked_scatter``)."""
    mask_i = mask_1d.long()
    source_idx = torch.cumsum(mask_i, 0) - 1
    source_idx = torch.clamp(source_idx, 0, source.shape[0] - 1)
    source_idx_2d = source_idx.unsqueeze(-1).expand_as(inputs_embeds_row)
    gathered_rows = torch.gather(source, 0, source_idx_2d)
    return torch.where(mask_1d.unsqueeze(-1), gathered_rows, inputs_embeds_row)


def _assert_pcc_one(golden: torch.Tensor, candidate: torch.Tensor, *, label: str) -> None:
    pcc = compute_pcc(golden, candidate)
    assert pcc == pytest.approx(1.0, abs=1e-5), f"{label}: PCC={pcc} (expected 1.0)"


def _mask_1d_for_inline(mask: torch.Tensor) -> torch.Tensor:
    """Row mask ``[S]`` from forge-style ``[S, 1]`` (or ``[S, H]`` row-constant) mask."""
    if mask.ndim == 1:
        return mask
    if mask.shape[-1] == 1:
        return mask.squeeze(-1)
    return mask[..., 0]


def _compare_native_decomp_inlined_on_capture(
    data: torch.Tensor, mask: torch.Tensor, source: torch.Tensor, *, label: str
) -> None:
    native = data.clone().masked_scatter(mask, source)
    decomp = masked_scatter_decomp(data.clone(), mask, source.clone())
    inlined = _merge_inlined_rowwise_prev_fork(
        data.clone(), _mask_1d_for_inline(mask), source
    )
    _assert_pcc_one(native, decomp, label=f"{label} native vs decomposition")
    _assert_pcc_one(native, inlined, label=f"{label} native vs inlined rowwise")
    if data.dtype == torch.float32:
        assert torch.equal(native, decomp)
        assert torch.equal(native, inlined)


@pytest.mark.nightly
@pytest.mark.single_device
@pytest.mark.record_test_properties(category=Category.OP_TEST)
def test_deepseek_ocr_masked_scatter_real_loader_inputs_native_vs_decomp(monkeypatch):
    """Uses ``ModelLoader`` sample (doc.png + preprocess) — same tensors as ``masked_scatter`` in forge."""
    from tests.torch.ops.test_deepseek_ocr_sanity import (
        _DeepseekOCRBeforeDecoder,
        _load_deepseek_ocr_model_and_inputs,
    )

    captures: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    orig = torch.Tensor.masked_scatter

    def _recorder(self, mask, source):
        captures.append((self.clone(), mask, source.clone()))
        return orig(self, mask, source)

    monkeypatch.setattr(torch.Tensor, "masked_scatter", _recorder)

    model, inputs, patches, image_ori = _load_deepseek_ocr_model_and_inputs()
    pre = _DeepseekOCRBeforeDecoder(model)
    with torch.no_grad():
        pre(
            input_ids=inputs["input_ids"],
            images=[(patches, image_ori)],
            images_seq_mask=inputs["images_seq_mask"],
            images_spatial_crop=inputs["images_spatial_crop"],
        )

    assert captures, (
        "expected at least one masked_scatter from real DeepSeek-OCR inputs "
        "(vision path + doc.png sample)"
    )
    for i, (data, mask, source) in enumerate(captures):
        _compare_native_decomp_inlined_on_capture(
            data, mask, source, label=f"capture[{i}]"
        )


@pytest.mark.push
@pytest.mark.record_test_properties(category=Category.OP_TEST)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("seq_len,hidden", [(128, 256), (277, 1280)])
def test_deepseek_merge_native_decomp_inlined_pcc_one(dtype, seq_len, hidden):
    gen = torch.Generator().manual_seed(0)
    data, mask_1d, source = _deepseek_style_mask_and_source(
        seq_len, hidden, dtype=dtype, generator=gen
    )

    native = _merge_native_main(data, mask_1d, source)
    decomp = _merge_decomposition(data, mask_1d, source)
    inlined = _merge_inlined_rowwise_prev_fork(data.clone(), mask_1d, source)

    _assert_pcc_one(native, decomp, label="native vs decomposition")
    _assert_pcc_one(native, inlined, label="native vs inlined rowwise")

    if dtype == torch.float32:
        assert torch.equal(native, decomp)
        assert torch.equal(native, inlined)


@pytest.mark.push
@pytest.mark.record_test_properties(category=Category.OP_TEST)
def test_deepseek_merge_batch_row_matches_native():
    """``inputs_embeds[idx]`` in forge is one batch row ``[S, H]``; also check ``[1,S,H]``."""
    gen = torch.Generator().manual_seed(1)
    seq_len, hidden = 64, 512
    data_2d, mask_1d, source = _deepseek_style_mask_and_source(
        seq_len, hidden, dtype=torch.float32, generator=gen
    )
    data_3d = data_2d.unsqueeze(0)

    native_2d = _merge_native_main(data_2d, mask_1d, source)
    decomp_3d = masked_scatter_decomp(
        data_3d.clone(),
        mask_1d.unsqueeze(0).unsqueeze(-1),
        source.clone(),
    )
    native_3d = data_3d.clone().masked_scatter(mask_1d.unsqueeze(0).unsqueeze(-1), source)

    _assert_pcc_one(native_3d, decomp_3d, label="[1,S,H] native vs decomp")
    assert torch.equal(native_2d, decomp_3d.squeeze(0))
    assert torch.equal(native_2d, native_3d.squeeze(0))
