# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""HunyuanImage 2.1 — HunyuanImageTransformer2DModel (MMDiT) component test."""

import pytest
import torch
import torch_xla
import torch_xla.runtime as xr
from infra import Framework, run_graph_test
from infra.utilities.torch_multichip_utils import get_mesh

from third_party.tt_forge_models.hunyuan_image_2_1.pytorch import (
    ModelLoader,
    ModelVariant,
)


@pytest.mark.skip(
    reason="model size 17.45B — won't fit on a single chip; sharded variant runs"
)
def test_transformer():
    _run(sharded=False)


@pytest.mark.xfail(
    reason="Out of Memory: Not enough space to allocate 234881024 B DRAM buffer across 12 banks, where each bank needs to store 19574784 B, but bank size is 1071821792 B - https://github.com/tenstorrent/tt-xla/issues/4780"
)
@pytest.mark.nightly
@pytest.mark.model_test
@pytest.mark.llmbox
@pytest.mark.testing
def test_transformer_sharded():
    _run(sharded=True)


def _run(sharded: bool):
    xr.set_device_type("TT")
    torch.manual_seed(42)

    loader = ModelLoader(ModelVariant.TRANSFORMER)
    model = loader.load_model(dtype_override=torch.float32)
    inputs = loader.load_inputs(dtype_override=torch.float32)

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
