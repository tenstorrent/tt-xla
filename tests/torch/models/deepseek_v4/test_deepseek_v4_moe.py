# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import numpy as np
import pytest
import torch
import torch_xla
import torch_xla.runtime as xr
from infra import Framework, run_graph_test
from infra.evaluators import ComparisonConfig, PccConfig
from infra.utilities.torch_multichip_utils import enable_spmd
from torch_xla.distributed.spmd import Mesh
from tt_torch.sparse_mlp import enable_sparse_mlp

from third_party.tt_forge_models.deepseek_v4.modified_model import (
    model_decode_opt as model,
)

from . import realistic_inputs, utils, weight_loader

NON_HASH_LAYER_ID = 10
HASH_LAYER_ID = 0


@pytest.mark.nightly
@pytest.mark.llmbox
@pytest.mark.parametrize("model_name", ["deepseek-ai/DeepSeek-V4-Flash"])
@pytest.mark.parametrize("batch_size", [32])
@pytest.mark.parametrize("seq_len", [1, 32])
def test_flash_moe(model_name, batch_size, seq_len):
    enable_spmd()
    xr.set_device_type("TT")

    args = weight_loader.load_config_args(model_name, False)
    block = model.Block(NON_HASH_LAYER_ID, args)
    block.ffn.load_state_dict(
        weight_loader.load_moe_state_dict(model_name, layer_id=NON_HASH_LAYER_ID)
    )

    mesh = utils.make_2d_mesh()
    enable_sparse_mlp(block, mesh=mesh.mesh_shape, cluster_axis=0, config=args)

    ffn = block.ffn

    _, hidden_states = realistic_inputs.get_realistic_inputs(
        model_name, layer_id=args.n_hash_layers, batch_size=batch_size, seq_len=seq_len
    )

    def moe_shard_spec(ffn, args, kwargs):
        shard_specs = {}
        # hidden_states: [batch, seq, dim] — batch on batch, dim on model
        shard_specs[args[0]] = ("batch", None, None)

        mlp = ffn.mlp if hasattr(ffn, "mlp") else ffn
        shard_specs[mlp.router.gate.weight] = (None, "model")
        shard_specs[mlp.experts.gate_proj] = (("batch", "model"), None, None)
        shard_specs[mlp.experts.up_proj] = (("batch", "model"), None, None)
        shard_specs[mlp.experts.down_proj] = (("batch", "model"), None, None)

        shared = getattr(ffn, "shared_experts", None)
        if shared is not None:
            shard_specs[shared.w1.weight] = ("model", None)
            shard_specs[shared.w3.weight] = ("model", None)
            shard_specs[shared.w2.weight] = (None, "model")

        return shard_specs

    run_graph_test(
        ffn,
        [hidden_states],
        framework=Framework.TORCH,
        mesh=mesh,
        shard_spec_fn=moe_shard_spec,
        comparison_config=utils.PCC_98,
    )


class _HashMoERunner(torch.nn.Module):
    """Adapter that lets the hash-routed a2a MoE consume (hidden_states,
    input_ids) as two explicit forward args. Stashes the flattened input_ids
    on the gate so the patched Gate/MoE forwards can retrieve them."""

    def __init__(self, ffn_wrapper: torch.nn.Module):
        super().__init__()
        self.ffn = ffn_wrapper

    def forward(self, hidden_states, input_ids):
        self.ffn.mlp.router.gate._ambient_input_ids = input_ids.flatten()
        return self.ffn(hidden_states)


@pytest.mark.nightly
@pytest.mark.llmbox
@pytest.mark.parametrize("model_name", ["deepseek-ai/DeepSeek-V4-Flash"])
@pytest.mark.parametrize("batch_size", [32])
@pytest.mark.parametrize("seq_len", [1, 32])
def test_flash_moe_hash(model_name, batch_size, seq_len):
    enable_spmd()
    xr.set_device_type("TT")

    args = weight_loader.load_config_args(model_name, False)
    block = model.Block(HASH_LAYER_ID, args)
    block.ffn.load_state_dict(
        weight_loader.load_moe_state_dict(model_name, layer_id=HASH_LAYER_ID)
    )

    mesh = utils.make_2d_mesh()
    enable_sparse_mlp(block, mesh=mesh.mesh_shape, cluster_axis=0, config=args)
    runner = _HashMoERunner(block.ffn)

    ffn = block.ffn

    input_ids, hidden_states = realistic_inputs.get_realistic_inputs(
        model_name, layer_id=args.n_hash_layers, batch_size=batch_size, seq_len=seq_len
    )

    def moe_hash_shard_spec(runner, args, kwargs):
        shard_specs = {}
        # hidden_states: [batch, seq, dim]
        shard_specs[args[0]] = ("batch", None, "model")
        # input_ids: [batch, seq]
        shard_specs[args[1]] = ("batch", None)

        mlp = runner.ffn.mlp
        shard_specs[mlp.router.gate.weight] = (None, "model")

        shard_specs[mlp.experts.gate_proj] = (("batch", "model"), None, None)
        shard_specs[mlp.experts.up_proj] = (("batch", "model"), None, None)
        shard_specs[mlp.experts.down_proj] = (("batch", "model"), None, None)

        shared = getattr(runner.ffn, "shared_experts", None)
        if shared is not None:
            shard_specs[shared.w1.weight] = ("model", None)
            shard_specs[shared.w3.weight] = ("model", None)
            shard_specs[shared.w2.weight] = (None, "model")

        return shard_specs

    run_graph_test(
        runner,
        [hidden_states, input_ids],
        framework=Framework.TORCH,
        mesh=mesh,
        shard_spec_fn=moe_hash_shard_spec,
        comparison_config=utils.PCC_99,
    )
