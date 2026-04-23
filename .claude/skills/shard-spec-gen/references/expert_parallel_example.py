# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
#
# Expert Parallelism (EP) shard-spec example — TO BE FILLED IN
#
# Strategy: assign contiguous blocks of experts to devices.
# Each device owns (num_experts / num_devices) experts.
# Token routing uses All-to-All communication (vs. All-Reduce in Megatron TP).
#
# When to prefer EP over Megatron on MoE:
#   - num_experts is cleanly divisible by num_devices
#   - experts are large enough that intra-expert weight sharding (Megatron) adds
#     too many CCL ops per expert
#   - routing overhead of All-to-All is acceptable vs. sharding overhead
#
# TODO: fill in <placeholders> once a concrete MoE model is targeted.

import numpy as np
import pytest
import torch
import torch_xla.runtime as xr
from infra import Framework, run_graph_test
from infra.evaluators.evaluation_config import ComparisonConfig, PccConfig
from infra.utilities.torch_multichip_utils import enable_spmd
from torch_xla.distributed.spmd import Mesh

# from transformers.models.<arch>.modeling_<arch> import <MoELayerClass>
from tests.utils import parametrize_arch

# from third_party.tt_forge_models.<model>.<task>.pytorch.loader import ModelLoader, ModelVariant


@pytest.mark.nightly
@parametrize_arch(["llmbox"])  # add "galaxy" if targeting 32-chip
@pytest.mark.parametrize("seq_len", [1024])
def test_<model>_moe_expert_parallel(seq_len, arch):
    xr.set_device_type("TT")

    # TODO: replace with ModelLoader or AutoConfig
    # loader = ModelLoader(variant=ModelVariant.<VARIANT>)
    # config = loader.load_config()
    # config.num_hidden_layers = 1
    # model = loader.load_model(dtype_override=torch.bfloat16)
    # moe_layer = model.model.layers[0].mlp

    enable_spmd()
    num_devices = xr.global_runtime_device_count()
    mesh_shape = (1, num_devices) if arch == "llmbox" else (4, num_devices // 4)
    # EP uses the "model" axis to partition experts, same mesh shape as Megatron.
    mesh = Mesh(np.arange(num_devices), mesh_shape, ("batch", "model"))

    # TODO: confirm num_experts from config
    # num_experts = config.num_experts          # e.g. 64
    # experts_per_device = num_experts // num_devices  # must divide evenly

    hidden_states = torch.randn(
        (mesh_shape[0], seq_len, 0),  # TODO: replace 0 with config.hidden_size
        dtype=torch.bfloat16,
    )

    def get_shard_spec(layer, args, kwargs):
        shard_specs = {}

        # Expert Parallelism: shard expert list along the "model" axis.
        # Each device receives a contiguous slice of experts.
        # The expert dimension is dim-0 of the stacked weight tensors when
        # the model stores experts as a single batched parameter, e.g.:
        #
        #   layer.experts.weight  shape: [num_experts, out_features, in_features]
        #
        # TODO: confirm whether experts are stored as:
        #   (a) a ModuleList of individual Linear layers  → iterate and shard each
        #   (b) a single batched weight tensor            → shard on dim 0

        # Pattern (a): ModuleList — assign experts to devices via the model axis
        # for i, expert in enumerate(layer.experts):
        #     shard_specs[expert.gate_proj.weight] = ("model", None)   # col-parallel within expert
        #     shard_specs[expert.up_proj.weight]   = ("model", None)
        #     shard_specs[expert.down_proj.weight] = (None, "model")   # row-parallel within expert
        # NOTE: for pure EP (no intra-expert TP), all expert weights on a device
        # are replicated along the model axis — routing selects the right device.
        # Pure EP shard spec (no intra-expert TP):
        #   shard_specs[expert.gate_proj.weight] = (None, None)  # fully local, no sharding
        # TODO: decide between EP-only vs EP+TP hybrid

        # Pattern (b): batched expert weight [num_experts, out, in]
        # shard_specs[layer.experts.weight] = ("model", None, None)

        # Shared expert (if present) — always replicated, not sharded
        # if hasattr(layer, "shared_expert") and layer.shared_expert is not None:
        #     shard_specs[layer.shared_expert.gate_proj.weight] = (None, None)
        #     shard_specs[layer.shared_expert.up_proj.weight]   = (None, None)
        #     shard_specs[layer.shared_expert.down_proj.weight] = (None, None)

        return shard_specs

    run_graph_test(
        moe_layer,  # TODO: replace with actual layer variable
        [hidden_states],
        framework=Framework.TORCH,
        comparison_config=ComparisonConfig(pcc=PccConfig(required_pcc=0.98)),
        mesh=mesh,
        shard_spec_fn=get_shard_spec,
    )
