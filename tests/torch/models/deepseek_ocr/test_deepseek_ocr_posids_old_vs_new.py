# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
CPU-only pytest: verify old (register_buffer) vs new (dynamic arange)
position_ids in CLIPVisionEmbeddings produce identical output, then
run the new implementation through run_op_test for PCC against TT device.
Includes a whole-model pretrained test using early_stop="before_super".
"""

import copy
import math

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from infra import Framework, run_op_test

from tests.infra.evaluators.evaluation_config import ComparisonConfig, PccConfig


def get_abs_pos(abs_pos, tgt_size):
    dim = abs_pos.size(-1)
    abs_pos_new = abs_pos.squeeze(0)
    cls_token, old_pos_embed = abs_pos_new[:1], abs_pos_new[1:]
    src_size = int(math.sqrt(abs_pos_new.shape[0] - 1))
    tgt_size = int(math.sqrt(tgt_size))
    dtype = abs_pos.dtype

    if src_size != tgt_size:
        old_pos_embed = (
            old_pos_embed.view(1, src_size, src_size, dim)
            .permute(0, 3, 1, 2)
            .contiguous()
        )
        old_pos_embed = old_pos_embed.to(torch.float32)
        new_pos_embed = F.interpolate(
            old_pos_embed,
            size=(tgt_size, tgt_size),
            mode="bicubic",
            antialias=True,
            align_corners=False,
        ).to(dtype)
        new_pos_embed = new_pos_embed.permute(0, 2, 3, 1)
        new_pos_embed = new_pos_embed.view(tgt_size * tgt_size, dim)
        vision_pos_embed = torch.cat([cls_token, new_pos_embed], dim=0)
        vision_pos_embed = vision_pos_embed.view(1, tgt_size * tgt_size + 1, dim)
        return vision_pos_embed
    else:
        return abs_pos


# -- Old implementation (register_buffer) --
class CLIPVisionEmbeddingsOld(nn.Module):
    def __init__(self, hidden_size=1024, image_size=224, patch_size=14, num_channels=3):
        super().__init__()
        self.embed_dim = hidden_size
        self.image_size = image_size
        self.patch_size = patch_size

        self.class_embedding = nn.Parameter(torch.randn(self.embed_dim))
        self.patch_embedding = nn.Conv2d(
            in_channels=num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=False,
        )

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches + 1
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)
        self.register_buffer(
            "position_ids", torch.arange(self.num_positions).expand((1, -1))
        )

    def forward(self, pixel_values, patch_embeds):
        batch_size = pixel_values.shape[0]
        if patch_embeds is not None:
            pass
        else:
            patch_embeds = self.patch_embedding(pixel_values)
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)

        class_embeds = self.class_embedding.expand(batch_size, 1, -1)
        embeddings = torch.cat([class_embeds, patch_embeds], dim=1)
        embeddings = embeddings + get_abs_pos(
            self.position_embedding(self.position_ids), embeddings.size(1)
        )
        return embeddings


# -- New implementation (dynamic arange from num_embeddings) --
class CLIPVisionEmbeddingsNew(nn.Module):
    def __init__(self, hidden_size=1024, image_size=224, patch_size=14, num_channels=3):
        super().__init__()
        self.embed_dim = hidden_size
        self.image_size = image_size
        self.patch_size = patch_size

        self.class_embedding = nn.Parameter(torch.randn(self.embed_dim))
        self.patch_embedding = nn.Conv2d(
            in_channels=num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=False,
        )

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches + 1
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)

    def forward(self, pixel_values, patch_embeds):
        batch_size = pixel_values.shape[0]
        if patch_embeds is not None:
            pass
        else:
            patch_embeds = self.patch_embedding(pixel_values)
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)

        class_embeds = self.class_embedding.expand(batch_size, 1, -1)
        embeddings = torch.cat([class_embeds, patch_embeds], dim=1)
        num_pos = self.position_embedding.num_embeddings
        position_ids = torch.arange(
            num_pos,
            device=embeddings.device,
            dtype=torch.long,
        ).unsqueeze(0)
        embeddings = embeddings + get_abs_pos(
            self.position_embedding(position_ids), embeddings.size(1)
        )
        return embeddings


def _copy_weights(src, dst):
    """Copy matching parameters from src to dst, skip missing keys."""
    dst_sd = dst.state_dict()
    for k, v in src.state_dict().items():
        if k in dst_sd:
            dst_sd[k] = v.clone()
    dst.load_state_dict(dst_sd, strict=False)


def _pearson_cc(a, b):
    a_flat = a.flatten().float()
    b_flat = b.flatten().float()
    a_c = a_flat - a_flat.mean()
    b_c = b_flat - b_flat.mean()
    return ((a_c * b_c).sum() / (a_c.norm() * b_c.norm())).item()


# --------------------------------------------------------------------------- #
# Test 1: old vs new produce identical output when sizes match (CPU only)
# --------------------------------------------------------------------------- #
@pytest.mark.single_device
def test_old_vs_new_matching_sizes_cpu():
    """When init num_positions == num_embeddings, old and new must be bit-exact."""
    torch.manual_seed(42)
    hidden, img, patch = 256, 56, 14

    old_model = CLIPVisionEmbeddingsOld(hidden, img, patch)
    new_model = CLIPVisionEmbeddingsNew(hidden, img, patch)
    _copy_weights(old_model, new_model)

    old_model.eval()
    new_model.eval()

    pixel_values = torch.randn(1, 3, img, img)

    with torch.no_grad():
        out_old = old_model(pixel_values, patch_embeds=None)
        out_new = new_model(pixel_values, patch_embeds=None)

    pcc = _pearson_cc(out_old, out_new)
    max_diff = (out_old - out_new).abs().max().item()

    print(f"[matching sizes] PCC = {pcc:.10f}  |  max abs diff = {max_diff:.2e}")
    assert pcc > 0.999999, f"PCC too low: {pcc}"
    assert max_diff < 1e-6, f"Max diff too large: {max_diff}"


# --------------------------------------------------------------------------- #
# Test 2: mismatched sizes -- old crashes, new survives (CPU only)
# --------------------------------------------------------------------------- #
@pytest.mark.single_device
def test_mismatched_sizes_old_crashes_new_survives_cpu():
    """
    Simulate the real bug: init gives 257 positions, but checkpoint embedding
    table has only 17 rows.  Old model must IndexError, new model must survive.
    """
    torch.manual_seed(42)
    hidden = 256
    crash_img, crash_patch = 224, 14  # num_positions = 257

    small_checkpoint_embed = torch.randn(17, hidden)

    old_model = CLIPVisionEmbeddingsOld(hidden, crash_img, crash_patch)
    new_model = CLIPVisionEmbeddingsNew(hidden, crash_img, crash_patch)

    # Directly replace the embedding module to simulate a checkpoint whose
    # position_embedding table has fewer rows than init computed.
    # (load_state_dict rejects shape mismatches even with strict=False.)
    old_model.position_embedding = nn.Embedding(17, hidden)
    old_model.position_embedding.weight = nn.Parameter(small_checkpoint_embed.clone())

    new_model.position_embedding = nn.Embedding(17, hidden)
    new_model.position_embedding.weight = nn.Parameter(small_checkpoint_embed.clone())

    pixel_values = torch.randn(1, 3, crash_img, crash_img)

    old_model.eval()
    new_model.eval()

    with pytest.raises(IndexError):
        with torch.no_grad():
            old_model(pixel_values, patch_embeds=None)

    with torch.no_grad():
        out_new = new_model(pixel_values, patch_embeds=None)
    assert out_new is not None
    print(f"[mismatched] New model output shape: {out_new.shape} -- PASS")


# --------------------------------------------------------------------------- #
# Test 3: run new implementation through run_op_test for PCC (CPU vs device)
# --------------------------------------------------------------------------- #
@pytest.mark.single_device
def test_new_clip_vision_embeddings_run_op_test():
    """
    Feed the new CLIPVisionEmbeddingsNew through run_op_test so the infra
    runs it on CPU and TT device and checks PCC automatically.
    """
    torch.manual_seed(42)
    hidden, img, patch = 256, 56, 14

    model = CLIPVisionEmbeddingsNew(hidden, img, patch)
    model.eval()

    pixel_values = torch.randn(1, 3, img, img)
    patch_embeds = None

    comparison_config = ComparisonConfig(
        pcc=PccConfig(required_pcc=0.99),
    )

    run_op_test(
        model,
        [pixel_values, patch_embeds],
        comparison_config=comparison_config,
        framework=Framework.TORCH,
    )
