# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Qwen-Image — QwenImageTransformer2DModel (transformer, ~20B) tensor-parallel component test.

The 40.9 GiB (bf16) MMDiT transformer exceeds every single TT chip, so it runs
Megatron 1D tensor-parallel on the llmbox fabric. 24 attention heads and 60
blocks divide cleanly across 2/3/4/6/8 devices.
"""

import pytest
import torch
import torch_xla
import torch_xla.runtime as xr
from infra import Framework, run_graph_test
from infra.utilities.torch_multichip_utils import get_mesh

from third_party.tt_forge_models.qwen_image.pytorch import ModelLoader, ModelVariant


@pytest.mark.nightly
@pytest.mark.model_test
@pytest.mark.llmbox
@pytest.mark.tensor_parallel
def test_transformer_sharded():
    xr.set_device_type("TT")
    torch.manual_seed(42)

    loader = ModelLoader(ModelVariant.QWEN_IMAGE_TRANSFORMER)
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
