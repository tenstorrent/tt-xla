# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""LTX-2 — Gemma3ForConditionalGeneration (text_encoder, ~12B) tensor-parallel component test."""

import pytest
import torch
import torch_xla
import torch_xla.runtime as xr
from infra import Framework, run_graph_test
from infra.utilities.torch_multichip_utils import get_mesh

from third_party.tt_forge_models.ltx2.pytorch import ModelLoader, ModelVariant


@pytest.mark.xfail(
    reason="Gemma3-12B sharded graph fails TT compile during "
    "_xla_warm_up_cache: 'ValueError: Error code: 13' "
    "(dynamo_bridge.extract_graph_helper). Tracing + fp32 CPU golden succeed; "
    "the 8-way sharded compile of the full multimodal Gemma3 does not. Distinct "
    "from the connectors/transformer rope issue — "
    "https://github.com/tenstorrent/tt-xla/issues/5197"
)
@pytest.mark.model_test
@pytest.mark.tensor_parallel
def test_text_encoder_sharded():
    xr.set_device_type("TT")
    torch.manual_seed(42)

    loader = ModelLoader(ModelVariant.LTX2_TEXT_ENCODER)
    model = loader.load_model(dtype_override=torch.bfloat16)
    inputs = loader.load_inputs(dtype_override=torch.bfloat16)

    mesh_shape, mesh_names = loader.get_mesh_config(xr.global_runtime_device_count())
    mesh = get_mesh(mesh_shape, mesh_names)

    run_graph_test(
        model,
        inputs,
        framework=Framework.TORCH,
        mesh=mesh,
        shard_spec_fn=loader.load_shard_spec,
    )
