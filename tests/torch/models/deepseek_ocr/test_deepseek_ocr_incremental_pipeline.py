# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Incremental DeepSeek-OCR **vision path** + scatter: CPU vs TT PCC to localize OOM / failures.

**Aligned with a real CPU run** (see ``deepseek_ocr_whole_model.log``):

- ``Getting input embeddings from input_ids``
- ``Processing images in forward pass, batch size is 1``
- ``Processing image 0 with crop shape tensor([2, 3])``  → ``width_crop_num=2``, ``height_crop_num=3``
- ``Processing patches for image 0`` → **patches branch** (``torch.sum(patches) != 0``), **not** the
  ``No patches for image …, only processing global features`` branch
- ``Concatenating features for image 0, num features: 1``

Tensor geometry (matches ``model_utils.preprocess`` + ``modeling_deepseekocr``):

- **patches**: stacked crops ``[N_crops, 3, 640, 640]`` with ``N_crops = width_crop_num * height_crop_num = 6``
- **image_ori**: global padded view ``[1, 3, 1024, 1024]`` (``base_size=1024``)
- **crop_shape** buffer ``[2, 3]`` for the local grid layout

Stages **1–5** grow the graph incrementally:

- **1–4**: individual vision components (SAM, CLIP, projector, global).
- **5a**: full vision stacking only (local + global + newline + separator → ``global_local_features``).
- **5b**: full vision stacking + scatter ops **through cumsum** only (no clamp / where / gather).
- **final**: full vision + **complete** scatter (broadcast → cumsum → clamp → gather → where → view_as).

Weights are **random init** (no HF checkpoint). Vision-only is not a separate stage: CLIP always
takes SAM outputs.
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn
import torch_xla.runtime as xr
from addict import Dict
from infra import Framework, run_op_test
from utils import Category

from third_party.tt_forge_models.deepseek.deepseek_ocr.pytorch.src.deepencoder import (
    MlpProjector,
    build_clip_l,
    build_sam_vit_b,
)

pytestmark = [pytest.mark.forked]

N_EMBED = 1280
# ``preprocess``: crop tiles ``image_size=640``; global ``base_size=1024``
PATCH_HW = 640
GLOBAL_HW = 1024
# From log: ``crop_shape`` / ``images_spatial_crop`` → grid 2×3 crops
CROP_W = 2
CROP_H = 3
N_CROPS = CROP_W * CROP_H
DTYPE = torch.bfloat16



def _whole_model_pixel_inputs(seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Same roles as ``image[0]`` / ``image[1]`` in ``DeepseekOCRModel.forward`` (patches branch)."""
    g = torch.Generator().manual_seed(seed)
    patches = torch.randn(N_CROPS, 3, PATCH_HW, PATCH_HW, dtype=DTYPE, generator=g)
    image_ori = torch.randn(1, 3, GLOBAL_HW, GLOBAL_HW, dtype=DTYPE, generator=g)
    return patches, image_ori


@pytest.fixture()
def vision_stack():
    """Fresh vision stack per test — ``run_op_test`` mutates models in-place via ``.to(device)``."""
    sam = build_sam_vit_b().eval().to(DTYPE)
    vision = build_clip_l().eval().to(DTYPE)
    projector = MlpProjector(
        Dict(projector_type="linear", input_dim=2048, n_embed=N_EMBED)
    ).eval().to(DTYPE)
    embed_scale = 1.0 / math.sqrt(float(N_EMBED))
    image_newline = nn.Parameter(torch.randn(N_EMBED, dtype=DTYPE) * embed_scale)
    view_seperator = nn.Parameter(torch.randn(N_EMBED, dtype=DTYPE) * embed_scale)
    return sam, vision, projector, image_newline, view_seperator


@pytest.fixture
def aligned_embeds_and_mask_for_final(vision_stack):
    """``images_seq_mask.sum()`` == number of rows in ``stacked_image_feats`` (required for scatter)."""
    patches, image_ori = _whole_model_pixel_inputs(99)
    sam, vision, proj, inl, sep = vision_stack
    # New ``Stage05FullStacked`` has CPU ``crop_shape`` buffer; ``.to(device)`` aligns buffers and
    # shared newline/separator params with ``sam``/``vision`` after earlier ``run_op_test`` runs.
    device = next(sam.parameters()).device
    stage5 = Stage05FullStacked(sam, vision, proj, inl, sep).to(device)
    patches_d = patches.to(device)
    image_ori_d = image_ori.to(device)
    with torch.no_grad():
        stacked = stage5(patches_d, image_ori_d)
    n_tok = int(stacked.shape[0])
    seq_len = n_tok + 10
    g = torch.Generator().manual_seed(100)
    inputs_embeds = torch.randn(seq_len, N_EMBED, dtype=DTYPE, generator=g)
    mask = torch.zeros(seq_len, dtype=torch.bool)
    mask[torch.randperm(seq_len, generator=g)[:n_tok]] = True
    assert int(mask.sum().item()) == n_tok
    # ``run_op_test`` expects CPU tensors like other stages (it moves inputs for TT).
    return inputs_embeds, mask, patches, image_ori


class Stage01SamPatchesOnly(nn.Module):
    def __init__(self, sam: nn.Module):
        super().__init__()
        self.sam = sam

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        return self.sam(patches)


class Stage02SamVisionLocal(nn.Module):
    def __init__(self, sam: nn.Module, vision: nn.Module):
        super().__init__()
        self.sam = sam
        self.vision = vision

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        lf1 = self.sam(patches)
        return self.vision(patches, lf1)


class Stage03LocalProjected(nn.Module):
    def __init__(self, sam: nn.Module, vision: nn.Module, projector: nn.Module):
        super().__init__()
        self.sam = sam
        self.vision = vision
        self.projector = projector

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        lf1 = self.sam(patches)
        lf2 = self.vision(patches, lf1)
        local_features = torch.cat(
            (lf2[:, 1:], lf1.flatten(2).permute(0, 2, 1)),
            dim=-1,
        )
        return self.projector(local_features)


class Stage04GlobalProjected(nn.Module):
    def __init__(self, sam: nn.Module, vision: nn.Module, projector: nn.Module):
        super().__init__()
        self.sam = sam
        self.vision = vision
        self.projector = projector

    def forward(self, image_ori: torch.Tensor) -> torch.Tensor:
        gf1 = self.sam(image_ori)
        gf2 = self.vision(image_ori, gf1)
        global_features = torch.cat(
            (gf2[:, 1:], gf1.flatten(2).permute(0, 2, 1)),
            dim=-1,
        )
        return self.projector(global_features)


class Stage05FullStacked(nn.Module):
    """``patches`` branch with ``crop_shape`` from whole-model log (``[2, 3]``)."""

    def __init__(
        self,
        sam: nn.Module,
        vision: nn.Module,
        projector: nn.Module,
        image_newline: nn.Parameter,
        view_seperator: nn.Parameter,
        crop_w: int = CROP_W,
        crop_h: int = CROP_H,
    ):
        super().__init__()
        self.sam = sam
        self.vision = vision
        self.projector = projector
        self.image_newline = image_newline
        self.view_seperator = view_seperator
        self.register_buffer(
            "crop_shape",
            torch.tensor([crop_w, crop_h], dtype=torch.long),
            persistent=False,
        )

    def forward(self, patches: torch.Tensor, image_ori: torch.Tensor) -> torch.Tensor:
        crop_shape = self.crop_shape
        width_crop_num, height_crop_num = int(crop_shape[0]), int(crop_shape[1])

        local_features_1 = self.sam(patches)
        local_features_2 = self.vision(patches, local_features_1)
        local_features = torch.cat(
            (
                local_features_2[:, 1:],
                local_features_1.flatten(2).permute(0, 2, 1),
            ),
            dim=-1,
        )
        local_features = self.projector(local_features)

        global_features_1 = self.sam(image_ori)
        global_features_2 = self.vision(image_ori, global_features_1)
        global_features = torch.cat(
            (
                global_features_2[:, 1:],
                global_features_1.flatten(2).permute(0, 2, 1),
            ),
            dim=-1,
        )
        global_features = self.projector(global_features)

        _, hw, n_dim = global_features.shape
        h = w = int(hw**0.5)
        _2, hw2, n_dim2 = local_features.shape
        h2 = w2 = int(hw2**0.5)

        global_features = global_features.view(h, w, n_dim)
        global_features = torch.cat(
            [
                global_features,
                self.image_newline[None, None, :].expand(h, 1, n_dim),
            ],
            dim=1,
        )
        global_features = global_features.view(-1, n_dim)

        local_features = (
            local_features.view(height_crop_num, width_crop_num, h2, w2, n_dim2)
            .permute(0, 2, 1, 3, 4)
            .reshape(height_crop_num * h2, width_crop_num * w2, n_dim2)
        )
        local_features = torch.cat(
            [
                local_features,
                self.image_newline[None, None, :].expand(height_crop_num * h2, 1, n_dim2),
            ],
            dim=1,
        )
        local_features = local_features.view(-1, n_dim2)

        global_local_features = torch.cat(
            [
                local_features,
                global_features,
                self.view_seperator[None, :],
            ],
            dim=0,
        )
        return global_local_features


class Stage05FullVisionThroughCumsum(nn.Module):
    """Full vision stacking + scatter broadcast/reshape/cumsum (no clamp/where/gather)."""

    def __init__(
        self,
        sam: nn.Module,
        vision: nn.Module,
        projector: nn.Module,
        image_newline: nn.Parameter,
        view_seperator: nn.Parameter,
    ):
        super().__init__()
        self.stack = Stage05FullStacked(
            sam, vision, projector, image_newline, view_seperator
        )

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        images_seq_mask: torch.Tensor,
        patches: torch.Tensor,
        image_ori: torch.Tensor,
    ) -> torch.Tensor:
        stacked_image_feats = self.stack(patches, image_ori)

        mask = images_seq_mask.unsqueeze(-1)
        mask_broad, data = torch.broadcast_tensors(mask, inputs_embeds)
        mask_flat = mask_broad.reshape(-1)
        source_flat = stacked_image_feats.reshape(-1)
        mask_i = mask_flat.long()
        source_idx = torch.cumsum(mask_i, 0) - 1

        return torch.cat([source_flat, source_idx.to(source_flat.dtype)])


class StageFullVisionPathPlusScatter(nn.Module):
    """Stage 5 (``global_local_features``) immediately followed by the full scatter block (``idx==0``)."""

    def __init__(
        self,
        sam: nn.Module,
        vision: nn.Module,
        projector: nn.Module,
        image_newline: nn.Parameter,
        view_seperator: nn.Parameter,
    ):
        super().__init__()
        self.stack = Stage05FullStacked(
            sam, vision, projector, image_newline, view_seperator
        )

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        images_seq_mask: torch.Tensor,
        patches: torch.Tensor,
        image_ori: torch.Tensor,
    ) -> torch.Tensor:
        stacked_image_feats = self.stack(patches, image_ori)
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
def test_stage_01_sam_patches_only_cpu_vs_tt_pcc(vision_stack):
    xr.set_device_type("TT")
    sam, *_ = vision_stack
    patches, _ = _whole_model_pixel_inputs(11)
    run_op_test(
        Stage01SamPatchesOnly(sam),
        [patches],
        framework=Framework.TORCH
    )


@pytest.mark.nightly
@pytest.mark.single_device
@pytest.mark.record_test_properties(category=Category.OP_TEST)
def test_stage_02_sam_vision_local_cpu_vs_tt_pcc(vision_stack):
    xr.set_device_type("TT")
    sam, vision, *_ = vision_stack
    patches, _ = _whole_model_pixel_inputs(12)
    run_op_test(
        Stage02SamVisionLocal(sam, vision),
        [patches],
        framework=Framework.TORCH
    )


@pytest.mark.nightly
@pytest.mark.single_device
@pytest.mark.record_test_properties(category=Category.OP_TEST)
def test_stage_03_local_projected_cpu_vs_tt_pcc(vision_stack):
    xr.set_device_type("TT")
    sam, vision, projector, *_ = vision_stack
    patches, _ = _whole_model_pixel_inputs(13)
    run_op_test(
        Stage03LocalProjected(sam, vision, projector),
        [patches],
        framework=Framework.TORCH
    )


@pytest.mark.nightly
@pytest.mark.single_device
@pytest.mark.record_test_properties(category=Category.OP_TEST)
def test_stage_04_global_projected_cpu_vs_tt_pcc(vision_stack):
    xr.set_device_type("TT")
    sam, vision, projector, *_ = vision_stack
    _, image_ori = _whole_model_pixel_inputs(14)
    run_op_test(
        Stage04GlobalProjected(sam, vision, projector),
        [image_ori],
        framework=Framework.TORCH
    )


@pytest.mark.nightly
@pytest.mark.single_device
@pytest.mark.record_test_properties(category=Category.OP_TEST)
def test_stage_05a_full_stacked_features_cpu_vs_tt_pcc(vision_stack):
    """Full vision stacking only (local + global + newline + separator)."""
    xr.set_device_type("TT")
    sam, vision, projector, image_newline, view_seperator = vision_stack
    patches, image_ori = _whole_model_pixel_inputs(15)
    run_op_test(
        Stage05FullStacked(sam, vision, projector, image_newline, view_seperator),
        [patches, image_ori],
        framework=Framework.TORCH,
    )


@pytest.mark.nightly
@pytest.mark.single_device
@pytest.mark.record_test_properties(category=Category.OP_TEST)
def test_stage_05b_full_vision_through_cumsum_cpu_vs_tt_pcc(
    vision_stack, aligned_embeds_and_mask_for_final
):
    """Full vision stacking + scatter ops through cumsum (no clamp/where/gather)."""
    xr.set_device_type("TT")
    inputs_embeds, mask, patches, image_ori = aligned_embeds_and_mask_for_final
    sam, vision, proj, inl, sep = vision_stack
    run_op_test(
        Stage05FullVisionThroughCumsum(sam, vision, proj, inl, sep),
        [inputs_embeds, mask, patches, image_ori],
        framework=Framework.TORCH,
    )


@pytest.mark.nightly
@pytest.mark.single_device
@pytest.mark.record_test_properties(category=Category.OP_TEST)
def test_stage_final_full_vision_path_plus_scatter_cpu_vs_tt_pcc(
    vision_stack, aligned_embeds_and_mask_for_final
):
    """Full vision stack → ``stacked_image_feats`` + scatter into ``inputs_embeds`` (single graph)."""
    xr.set_device_type("TT")
    inputs_embeds, mask, patches, image_ori = aligned_embeds_and_mask_for_final
    sam, vision, proj, inl, sep = vision_stack
    run_op_test(
        StageFullVisionPathPlusScatter(sam, vision, proj, inl, sep),
        [inputs_embeds, mask, patches, image_ori],
        framework=Framework.TORCH
    )
