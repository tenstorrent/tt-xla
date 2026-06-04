# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Mochi — MochiTransformer3DModel (DiT) component test (10.03B)."""

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
    load_transformer_inputs_full_res,
)


@pytest.mark.xfail(
    reason="Out of Memory: Not enough space to allocate 32463388672 B DRAM buffer "
    "across 12 banks, where each bank needs to store 2705283072 B, but bank size "
    "is 1071821792 B — https://github.com/tenstorrent/tt-xla/issues/4651"
)
def test_transformer_sharded():
    # Not using run_graph_test: it runs the model on CPU as a golden reference
    # first, and the 10B transformer CPU pass takes ~12 min at full resolution.
    # TT-only for now.
    # TODO: re-enable CPU golden + PCC check via run_graph_test once the TT
    # path is passing - https://github.com/tenstorrent/tt-xla/issues/4884
    torch_xla.set_custom_compile_options(
        {"experimental-enable-dram-space-saving-optimization": "true"}
    )
    xr.set_device_type("TT")
    os.environ["CONVERT_SHLO_TO_SHARDY"] = "1"
    xr.use_spmd()
    torch.manual_seed(42)

    device = xm.xla_device()

    loader = ModelLoader(ModelVariant.MOCHI, subfolder="transformer")
    model = loader.load_model(dtype_override=torch.bfloat16).eval().to(device)

    compiled = torch.compile(model, backend="tt")

    mesh_shape, mesh_names = loader.get_mesh_config(xr.global_runtime_device_count())
    mesh = get_mesh(mesh_shape, mesh_names)
    shard_spec = loader.load_shard_spec(model)
    for tensor, partition_spec in shard_spec.items():
        xs.mark_sharding(tensor, mesh, partition_spec)

    hidden_states, encoder_hidden_states, timestep, encoder_attention_mask = (
        load_transformer_inputs_full_res(dtype=torch.bfloat16)
    )
    hidden_states = hidden_states.to(device)
    encoder_hidden_states = encoder_hidden_states.to(device)
    timestep = timestep.to(device)
    encoder_attention_mask = encoder_attention_mask.to(device)

    with torch.no_grad():
        tt_out = compiled(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            timestep=timestep,
            encoder_attention_mask=encoder_attention_mask,
        )
