# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""HunyuanVideo 1.5 — AutoencoderKLHunyuanVideo15 decoder, 4-chip sharded.

Step 1 of sharding_analysis.md: shard the *activations* of the three large
upsamplers on H across the "model" mesh axis, keep every weight replicated.

Why: the DC-AE pixel shuffle in HunyuanVideo15Upsample (view -> permute ->
reshape) materialises an 8-D tensor whose last dim is 2. On one Blackhole chip
that tensor costs 28.3 GB at up_blocks[3] in either TILE or row-major layout
(tt-mlir #7729 / #8568, tt-xla Jira update 2026-09-07). Channel sharding of the
upsampler conv does not help: the packed channel dim is ordered (r1, r2, r3, c)
with c innermost, so a contiguous quarter is one spatial phase and Shardy has to
gather it back before the merge reshape. H is a *major* factor of the merged 2H
dim, so an H-shard survives view/permute/reshape and the permute output becomes
[1, C, F, 1, H/4, 2, W, 2] per device (7.1 GB at up_blocks[3]).

Per upsampler (only where H divides by the axis size: up_blocks[1..3]):
  1. upsampler pre-hook : x -> replicated anchor -> H-shard   (shortcut path runs on H/4)
  2. conv pre-hook      : x -> replicated                     (conv needs full spatial extent, no halo support: tt-mlir #8216)
  3. conv forward-hook  : h -> replicated anchor -> H-shard   (h path runs on H/4)
  4. upsampler fwd-hook : out -> replicated                   (next resnet's convs see a replicated input)
Anchor-then-shard pairs follow tests/torch/models/wan14b/shared.py::apply_dit_sp_activation_sharding
(stops Shardy back-propagating the shard through upstream reshapes).
"""

import torch
import torch_xla.runtime as xr
from infra import Framework, run_graph_test
from infra.testers.single_chip.model.torch_model_tester import _mask_jax_accelerator
from infra.utilities import Mesh
from infra.utilities.torch_multichip_utils import get_mesh
from loguru import logger
from tests.infra.testers.compiler_config import CompilerConfig
from tt_torch.sharding import sharding_constraint_tensor

from third_party.tt_forge_models.hunyuan_1_5.pytorch import ModelLoader, ModelVariant
from third_party.tt_forge_models.hunyuan_1_5.pytorch.src.model_utils import (
    MESH_NAMES,
    MESH_SHAPES,
)

SHARD_AXIS = "model"
H_DIM = 3  # activations are [B, C, F, H, W]


def _axis_size(mesh: Mesh, axis: str) -> int:
    return dict(zip(mesh.axis_names, mesh.mesh_shape))[axis]


def _shardable(t: torch.Tensor, n: int) -> bool:
    return t.ndim == 5 and t.shape[H_DIM] % n == 0


def _replicated(t: torch.Tensor, mesh: Mesh) -> torch.Tensor:
    return sharding_constraint_tensor(t, mesh, (None,) * t.ndim)


def _h_sharded(t: torch.Tensor, mesh: Mesh) -> torch.Tensor:
    spec = [None] * t.ndim
    spec[H_DIM] = SHARD_AXIS
    return sharding_constraint_tensor(t, mesh, tuple(spec))


def apply_upsampler_h_sharding(vae, mesh: Mesh) -> list[str]:
    """Register the four hooks on every decoder upsampler. Returns the names hooked.

    The divisibility check runs on the traced shapes, so up_blocks[0]
    (H=30 on the 480p latent) is left replicated automatically.
    """
    n = _axis_size(mesh, SHARD_AXIS)
    hooked = []

    def upsampler_pre(module, args):
        x = args[0]
        if not _shardable(x, n):
            return None
        x = _h_sharded(_replicated(x, mesh), mesh)
        return (x,) + tuple(args[1:])

    def conv_pre(module, args):
        x = args[0]
        if not _shardable(x, n):
            return None
        return (_replicated(x, mesh),) + tuple(args[1:])

    def conv_post(module, args, output):
        if not _shardable(output, n):
            return None
        return _h_sharded(_replicated(output, mesh), mesh)

    def upsampler_post(module, args, output):
        if not _shardable(output, n):
            return None
        return _replicated(output, mesh)

    for i, up_block in enumerate(vae.decoder.up_blocks):
        for j, upsampler in enumerate(up_block.upsamplers or []):
            upsampler.register_forward_pre_hook(upsampler_pre)
            upsampler.register_forward_hook(upsampler_post)
            upsampler.conv.register_forward_pre_hook(conv_pre)
            upsampler.conv.register_forward_hook(conv_post)
            hooked.append(f"decoder.up_blocks[{i}].upsamplers[{j}]")
    return hooked


def shard_vae_decoder_specs_step1(model) -> dict:
    """Weight specs for Step 1: everything replicated.

    One explicit replicated annotation is kept so the graph carries the mesh
    even though no weight is split (harmless; same form Wan uses for its
    replicated boundary weights).
    """
    conv_in = model.vae.decoder.conv_in.conv
    return {conv_in.weight: (None, None, None, None, None)}


def test_vae_decoder_sharded():
    xr.set_device_type("TT")
    torch.manual_seed(42)

    loader = ModelLoader(ModelVariant.VAE)
    model = loader.load_model(dtype_override=torch.bfloat16)
    inputs = loader.load_inputs(dtype_override=torch.bfloat16)

    num_devices = xr.global_runtime_device_count()
    mesh = get_mesh(MESH_SHAPES[num_devices], MESH_NAMES)

    # Hooks live on the modules, so they survive the runner's model.to(device).
    hooked = apply_upsampler_h_sharding(model.vae, mesh)
    logger.info("H-sharded upsamplers on axis '{}': {}", SHARD_AXIS, hooked)

    with _mask_jax_accelerator():
        run_graph_test(
            model,
            inputs,
            framework=Framework.TORCH,
            mesh=mesh,
            shard_spec_fn=shard_vae_decoder_specs_step1,
            # Required regardless of sharding: runs the pixel-shuffle permute and
            # reshape in row-major (tt-mlir #7729 / #8568).
            compiler_config=CompilerConfig(
                experimental_enable_dram_space_saving_optimization=True,
            ),
        )
