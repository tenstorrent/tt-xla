# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""OmniGen — AutoencoderKL encoder component test."""

import pytest
import torch
import torch_xla
import torch_xla.runtime as xr
from infra import Framework, run_graph_test
from infra.testers.single_chip.model.torch_model_tester import _mask_jax_accelerator
from infra.utilities.torch_multichip_utils import get_mesh

from third_party.tt_forge_models.omnigen.pytorch import ModelLoader, ModelVariant


@pytest.mark.skip
def test_vae_encoder():
    _run(sharded=False)


@pytest.mark.nightly
@pytest.mark.model_test
@pytest.mark.llmbox
@pytest.mark.testing
def test_vae_encoder_sharded():
    _run(sharded=True)


def _run(sharded: bool):
    xr.set_device_type("TT")
    torch.manual_seed(42)

    loader = ModelLoader(ModelVariant.VAE_ENCODER)
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

    with _mask_jax_accelerator():
        run_graph_test(
            model,
            inputs,
            framework=Framework.TORCH,
            mesh=mesh,
            shard_spec_fn=shard_spec_fn,
        )
