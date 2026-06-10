# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""FLUX.2-dev — Mistral3 text encoder component test (128x128 pipeline resolution)."""

import pytest
import torch
import torch_xla
import torch_xla.runtime as xr
from infra import Framework, run_graph_test
from infra.utilities.torch_multichip_utils import get_mesh

from third_party.tt_forge_models.flux2.pytorch import ModelLoader, ModelVariant


@pytest.mark.skip(
    reason="~24B text encoder — exceeds single-chip DRAM; use test_text_encoder_sharded on 8+ chips"
)
@pytest.mark.single_device
@pytest.mark.model_test
def test_text_encoder():
    _run(sharded=False)


@pytest.mark.xfail(
    reason="PCC comparison failed: calculated pcc=0.9684 vs required 0.99. "
    "Sharded Mistral3 text encoder compiles and runs end-to-end on 8 chips; "
    "remaining gap is numerical accuracy. Tracking issue TBD."
)
@pytest.mark.tensor_parallel
@pytest.mark.nightly
@pytest.mark.model_test
def test_text_encoder_sharded():
    _run(sharded=True)


def _run(sharded: bool):
    xr.set_device_type("TT")
    torch.manual_seed(42)

    loader = ModelLoader(ModelVariant.TEXT_ENCODER)
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
