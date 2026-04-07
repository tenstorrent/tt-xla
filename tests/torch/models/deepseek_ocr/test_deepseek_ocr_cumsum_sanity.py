# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-OCR masked_scatter decomposition — five sanities, **CPU vs TT PCC** (no ``masked_scatter_``).

Same shapes/dtypes as ``inputs_embeds[idx]`` / ``images_seq_mask[idx]`` in ``DeepseekOCRModel.forward``.
No third-party loading.

**1.** Cumsum only: ``source_idx = torch.cumsum(mask_i, 0) - 1``

**2.** Broadcast + ``mask_flat`` + ``mask_i`` + cumsum

**3.** + ``source_flat``, ``clamp``, ``gathered = source_flat[source_idx]``

**4.** + ``data_flat``, ``torch.where(mask_flat, gathered, data_flat)``

**5.** + ``result_flat.view_as(inputs_embeds)``
"""

import pytest
import torch
import torch.nn as nn
import torch_xla.runtime as xr
from infra import Framework, run_op_test
from utils import Category


HIDDEN_SIZE = 1280
SEQ_LEN = 512
MASK_I_NUMEL = SEQ_LEN * HIDDEN_SIZE
INPUTS_EMBEDS_DTYPE = torch.bfloat16


def _make_row_inputs(seed: int):
    g = torch.Generator().manual_seed(seed)
    inputs_embeds = torch.randn(
        SEQ_LEN, HIDDEN_SIZE, dtype=INPUTS_EMBEDS_DTYPE, generator=g
    )
    images_seq_mask = torch.zeros(SEQ_LEN, dtype=torch.bool)
    images_seq_mask[torch.randperm(SEQ_LEN, generator=g)[:200]] = True
    n_img = int(images_seq_mask.sum().item())
    stacked_image_feats = torch.randn(
        n_img, HIDDEN_SIZE, dtype=INPUTS_EMBEDS_DTYPE, generator=g
    )
    return inputs_embeds, images_seq_mask, stacked_image_feats


def _make_mask_i(seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, 2, (MASK_I_NUMEL,), dtype=torch.int64, generator=g)


class Sanity01CumsumOnly(nn.Module):
    def forward(self, mask_i: torch.Tensor) -> torch.Tensor:
        return (torch.cumsum(mask_i, 0) - 1).float()


class Sanity02BroadcastCumsum(nn.Module):
    def forward(
        self, inputs_embeds: torch.Tensor, images_seq_mask: torch.Tensor
    ) -> torch.Tensor:
        mask = images_seq_mask.unsqueeze(-1)
        mask_broad, _data = torch.broadcast_tensors(mask, inputs_embeds)
        mask_flat = mask_broad.reshape(-1)
        mask_i = mask_flat.long()
        return (torch.cumsum(mask_i, 0) - 1).float()


class Sanity03ClampGather(nn.Module):
    def forward(
        self,
        inputs_embeds: torch.Tensor,
        images_seq_mask: torch.Tensor,
        stacked_image_feats: torch.Tensor,
    ) -> torch.Tensor:
        mask = images_seq_mask.unsqueeze(-1)
        mask_broad, _data = torch.broadcast_tensors(mask, inputs_embeds)
        mask_flat = mask_broad.reshape(-1)
        source_flat = stacked_image_feats.reshape(-1)
        mask_i = mask_flat.long()
        source_idx = torch.cumsum(mask_i, 0) - 1
        source_idx = torch.clamp(source_idx, 0, source_flat.shape[0] - 1)
        return source_flat[source_idx]


class Sanity04Where(nn.Module):
    def forward(
        self,
        inputs_embeds: torch.Tensor,
        images_seq_mask: torch.Tensor,
        stacked_image_feats: torch.Tensor,
    ) -> torch.Tensor:
        mask = images_seq_mask.unsqueeze(-1)
        mask_broad, data = torch.broadcast_tensors(mask, inputs_embeds)
        mask_flat = mask_broad.reshape(-1)
        data_flat = data.reshape(-1)
        source_flat = stacked_image_feats.reshape(-1)
        mask_i = mask_flat.long()
        source_idx = torch.cumsum(mask_i, 0) - 1
        source_idx = torch.clamp(source_idx, 0, source_flat.shape[0] - 1)
        gathered = source_flat[source_idx]
        return torch.where(mask_flat, gathered, data_flat)


class Sanity05ViewAs(nn.Module):
    def forward(
        self,
        inputs_embeds: torch.Tensor,
        images_seq_mask: torch.Tensor,
        stacked_image_feats: torch.Tensor,
    ) -> torch.Tensor:
        mask = images_seq_mask.unsqueeze(-1)
        mask_broad, data = torch.broadcast_tensors(mask, inputs_embeds)
        mask_flat = mask_broad.reshape(-1)
        data_flat = data.reshape(-1)
        source_flat = stacked_image_feats.reshape(-1)
        mask_i = mask_flat.long()
        source_idx = torch.cumsum(mask_i, 0) - 1
        source_idx = torch.clamp(source_idx, 0, source_flat.shape[0] - 1)
        gathered = source_flat[source_idx]
        result_flat = torch.where(mask_flat, gathered, data_flat)
        return result_flat.view_as(inputs_embeds)


@pytest.mark.nightly
@pytest.mark.single_device
@pytest.mark.record_test_properties(category=Category.OP_TEST)
def test_sanity_01_cumsum_only_cpu_vs_tt_pcc():
    xr.set_device_type("TT")
    mask_i = _make_mask_i(0)
    run_op_test(
        Sanity01CumsumOnly(),
        [mask_i],
        framework=Framework.TORCH
    )


@pytest.mark.nightly
@pytest.mark.single_device
@pytest.mark.record_test_properties(category=Category.OP_TEST)
def test_sanity_02_broadcast_cumsum_cpu_vs_tt_pcc():
    xr.set_device_type("TT")
    inputs_embeds, images_seq_mask, _s = _make_row_inputs(1)
    run_op_test(
        Sanity02BroadcastCumsum(),
        [inputs_embeds, images_seq_mask],
        framework=Framework.TORCH
    )


@pytest.mark.nightly
@pytest.mark.single_device
@pytest.mark.record_test_properties(category=Category.OP_TEST)
def test_sanity_03_clamp_gather_cpu_vs_tt_pcc():
    xr.set_device_type("TT")
    inputs_embeds, images_seq_mask, stacked = _make_row_inputs(2)
    run_op_test(
        Sanity03ClampGather(),
        [inputs_embeds, images_seq_mask, stacked],
        framework=Framework.TORCH
    )


@pytest.mark.nightly
@pytest.mark.single_device
@pytest.mark.record_test_properties(category=Category.OP_TEST)
def test_sanity_04_where_cpu_vs_tt_pcc():
    xr.set_device_type("TT")
    inputs_embeds, images_seq_mask, stacked = _make_row_inputs(3)
    run_op_test(
        Sanity04Where(),
        [inputs_embeds, images_seq_mask, stacked],
        framework=Framework.TORCH
    )


@pytest.mark.nightly
@pytest.mark.single_device
@pytest.mark.record_test_properties(category=Category.OP_TEST)
def test_sanity_05_view_as_cpu_vs_tt_pcc():
    xr.set_device_type("TT")
    inputs_embeds, images_seq_mask, stacked = _make_row_inputs(4)
    run_op_test(
        Sanity05ViewAs(),
        [inputs_embeds, images_seq_mask, stacked],
        framework=Framework.TORCH
    )
