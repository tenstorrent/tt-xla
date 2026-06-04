# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Mochi VAE decoder component test at original Mochi-1 resolution (0.36B)."""

import os

import pytest
import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.spmd as xs
import torch_xla.runtime as xr
from infra.utilities.torch_multichip_utils import get_mesh

from third_party.tt_forge_models.mochi.pytorch import ModelLoader, ModelVariant
from third_party.tt_forge_models.mochi.pytorch.src.utils import (
    load_vae_decoder_inputs_full_res,
)


@pytest.mark.xfail(
    reason="DRAM OOM during sharded decoder pixel-shuffle intermediate — "
    "https://github.com/tenstorrent/tt-xla/issues/4252"
)
def test_torch_mochi_vae_decoder_inference():
    # Not using run_graph_test: it runs the model on CPU as a golden
    # reference first, and the Mochi VAE decoder CPU pass takes > 50 min at
    # full resolution. TT-only for now.
    # TODO: re-enable CPU golden + PCC check via run_graph_test once the
    # TT path is passing - https://github.com/tenstorrent/tt-xla/issues/4885
    # SPMD setup must come before any tensor lands on the XLA device.
    torch_xla.set_custom_compile_options(
        {"experimental-enable-dram-space-saving-optimization": "true"}
    )
    xr.set_device_type("TT")
    os.environ["CONVERT_SHLO_TO_SHARDY"] = "1"
    xr.use_spmd()
    torch.manual_seed(42)

    device = xm.xla_device()

    loader = ModelLoader(ModelVariant.MOCHI, subfolder="vae")
    vae = loader.load_model(dtype_override=torch.bfloat16)
    decoder = vae.decoder.eval().to(device)

    compiled = torch.compile(decoder, backend="tt")

    # Megatron-style sharding applied after weights are on the XLA device.
    mesh_shape, mesh_names = loader.get_mesh_config(xr.global_runtime_device_count())
    mesh = get_mesh(mesh_shape, mesh_names)
    shard_spec = loader.load_shard_spec(decoder)
    for tensor, partition_spec in shard_spec.items():
        xs.mark_sharding(tensor, mesh, partition_spec)

    [latent] = load_vae_decoder_inputs_full_res(dtype=torch.bfloat16)
    tt_latent = latent.to(device)

    with torch.no_grad():
        tt_out = compiled(tt_latent)
