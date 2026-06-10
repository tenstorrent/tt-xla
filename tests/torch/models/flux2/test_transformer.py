# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""FLUX.2-dev — Flux2Transformer2DModel component test (128x128 latent geometry)."""

import pytest
import torch
import torch_xla
import torch_xla.runtime as xr
from infra import Framework, run_graph_test
from infra.utilities.torch_multichip_utils import get_mesh

from third_party.tt_forge_models.flux2.pytorch import ModelLoader, ModelVariant


@pytest.mark.skip(
    reason="~32B transformer — exceeds single-chip DRAM; use test_transformer_sharded on 8+ chips"
)
@pytest.mark.single_device
@pytest.mark.model_test
def test_transformer():
    _run(sharded=False)


@pytest.mark.xfail(
    reason="Out of Memory: cannot allocate 56623104 B DRAM buffer across 12 banks "
    "(DRAM ~full, ~38 MB free) when sharded across 8 chips — "
    "TT_FATAL bank_manager.cpp:462. 32B transformer is still DRAM-bound. "
    "Tracking issue TBD."
)
@pytest.mark.tensor_parallel
@pytest.mark.nightly
@pytest.mark.model_test
def test_transformer_sharded():
    _run(sharded=True)


def _run(sharded: bool):
    xr.set_device_type("TT")
    torch.manual_seed(42)

    loader = ModelLoader(ModelVariant.TRANSFORMER)
    model = loader.load_model(dtype_override=torch.bfloat16)
    inputs = loader.load_inputs(dtype_override=torch.bfloat16)

    mesh = None
    shard_spec_fn = None
    if sharded:
        mesh_shape, mesh_names = loader.get_mesh_config(
            xr.global_runtime_device_count()
        )
        mesh = get_mesh(mesh_shape, mesh_names)
        shard_spec_fn = loader.load_shard_spec

    run_graph_test(
        model,
        inputs,
        framework=Framework.TORCH,
        mesh=mesh,
        shard_spec_fn=shard_spec_fn,
    )
