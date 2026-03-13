# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import torch
import torch_xla.runtime as xr
from infra import Framework, run_graph_test
from torch_xla.distributed.spmd import Mesh
from transformers.models.glm4_moe.modeling_glm4_moe import Glm4MoeMoE
from tt_torch.sparse_mlp import create_a2a_from_glm4_moe

from tests.utils import parametrize_arch
from third_party.tt_forge_models.glm.causal_lm.pytorch.loader import (
    ModelLoader as GLMModelLoader,
)

MODEL_LOADER_MAP = {
    "glm": GLMModelLoader,
}

AVAILABLE_VARIANT_MAP = {
    "glm": ["4.7", "4.5", "4.5_Air"],
}


def get_available_variants(model_name):
    ModelLoader = MODEL_LOADER_MAP[model_name]
    available_variants = ModelLoader.query_available_variants()

    # Filter to only include variants that match names in AVAILABLE_VARIANT_MAP
    if model_name in AVAILABLE_VARIANT_MAP:
        allowed_variant_names = set(AVAILABLE_VARIANT_MAP[model_name])
        return {
            variant_key: variant_config
            for variant_key, variant_config in available_variants.items()
            if str(variant_key) in allowed_variant_names
        }

    return available_variants


"""GLM MoE test"""


@pytest.mark.nightly
@parametrize_arch(["single_device", "llmbox"])
@pytest.mark.parametrize("seq_len", [1024])
@pytest.mark.parametrize(
    "variant,variant_config",
    get_available_variants("glm").items(),
    ids=[str(k) for k in get_available_variants("glm").keys()],
)
def test_glm_moe(variant, variant_config, seq_len, arch):
    xr.set_device_type("TT")

    loader = GLMModelLoader(variant=variant)
    config = loader.load_config()

    if arch == "llmbox":
        num_devices = xr.global_runtime_device_count()
        mesh_shape = (1, num_devices)
        device_ids = np.array(range(num_devices))
        mesh = Mesh(device_ids, mesh_shape, ("batch", "model"))
        # Dispatch along the model (column) axis so all-to-all routes tokens across devices
        cluster_axis = 1
    else:
        num_devices = 1
        mesh = None
        cluster_axis = 0

    glm_moe = Glm4MoeMoE(config).to(torch.bfloat16)
    moe = create_a2a_from_glm4_moe(
        glm_moe, config, num_devices=num_devices, cluster_axis=cluster_axis
    )

    batch_size = 4
    hidden_states = torch.randn(
        (batch_size, seq_len, config.hidden_size), dtype=torch.bfloat16
    )

    if arch == "llmbox":

        def get_shard_spec(model, args, kwargs):
            # Shard stacked expert weights along the expert (E) dimension across the model axis
            return {
                model.mlp.experts.gate_up_proj: ("model", None, None),
                model.mlp.experts.gate_up_proj_bias: ("model", None),
                model.mlp.experts.down_proj: ("model", None, None),
                model.mlp.experts.down_proj_bias: ("model", None),
            }

    else:
        get_shard_spec = None

    run_graph_test(
        moe,
        [hidden_states],
        framework=Framework.TORCH,
        mesh=mesh,
        shard_spec_fn=get_shard_spec,
    )
