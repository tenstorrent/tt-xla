# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
#
# CONVENTIONS:
# - xr.set_device_type("TT") must be first
# - enable_spmd() before mesh creation
# - torch.bfloat16 throughout
# - @parametrize_arch: use llmbox and/or galaxy — never single_device
# - batch size must equal mesh_shape[0] (the batch axis of the mesh)
# - one layer only — no stacking, no full forward pass
# - if using AutoConfig or manual config, add a comment with the correct HF model ID
# - theoretical mode: add "# WARNING: generated in theoretical mode — attribute paths and shard specs are unverified"

import numpy as np
import pytest
import torch
import torch_xla.runtime as xr
from infra import Framework, run_graph_test
from infra.evaluators.evaluation_config import ComparisonConfig, PccConfig
from infra.utilities.torch_multichip_utils import enable_spmd
from torch_xla.distributed.spmd import Mesh
from transformers.models.<arch>.modeling_<arch> import <LayerClass>

from tests.utils import parametrize_arch
from third_party.tt_forge_models.<model>.<task>.pytorch.loader import (
    ModelLoader,
    ModelVariant,
)


@pytest.mark.nightly
@parametrize_arch(["llmbox"])  # add "galaxy" if targeting 32-chip
@pytest.mark.parametrize("seq_len", [1024])
def test_<model>_<component>_sharded(seq_len, arch):
    xr.set_device_type("TT")

    loader = ModelLoader(variant=ModelVariant.<VARIANT>)
    config = loader.load_config()

    # attention/MLP: instantiate layer directly
    config._attn_implementation = "sdpa"  # attention only
    layer = <LayerClass>(config, layer_idx=0).to(torch.bfloat16)

    # MoE: load minimal model instead
    # config.num_hidden_layers = 1
    # model = loader.load_model(dtype_override=torch.bfloat16)
    # layer = model.model.layers[0].mlp

    enable_spmd()
    num_devices = xr.global_runtime_device_count()
    mesh_shape = (1, num_devices) if arch == "llmbox" else (4, num_devices // 4)
    mesh = Mesh(np.arange(num_devices), mesh_shape, ("batch", "model"))

    hidden_states = torch.randn(
        (mesh_shape[0], seq_len, config.hidden_size), dtype=torch.bfloat16
    )
    # Attention also needs position_embeddings, attention_mask, past_key_states:
    # head_dim = config.hidden_size // config.num_attention_heads
    # cos_sin = torch.rand(mesh_shape[0], seq_len, head_dim, dtype=torch.bfloat16)
    # position_embeddings = (cos_sin, cos_sin)
    # attention_mask = torch.rand(mesh_shape[0], 1, seq_len, seq_len, dtype=torch.bfloat16)
    # past_key_value = None

    def get_shard_spec(layer, args, kwargs):
        shard_specs = {}
        # populate from references/sharding_rules.md using confirmed attribute paths
        return shard_specs

    run_graph_test(
        layer,
        [hidden_states],
        framework=Framework.TORCH,
        comparison_config=ComparisonConfig(pcc=PccConfig(required_pcc=0.98)),
        mesh=mesh,
        shard_spec_fn=get_shard_spec,
    )
