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
from torch_xla.distributed.spmd import Mesh
from tt_torch.sparse_mlp import enable_sparse_mlp

from third_party.tt_forge_models.deepseek_v4.modified_model import model as ds_model

from . import realistic_inputs, weight_loader

# Layer past `n_hash_layers` so Gate uses score-based routing (no input_ids
# dependency). Layer 10 is a typical routed-MoE layer in V4-Flash (n_hash=3).
LAYER_ID = 10
# Layer < n_hash_layers uses static hash routing via gate.tid2eid[input_ids].
HASH_LAYER_ID = 0


@pytest.mark.nightly
@pytest.mark.llmbox
@pytest.mark.parametrize("batch_size", [32])
@pytest.mark.parametrize("seq_len", [1, 32])
def test_deepseek_v4_flash_moe(batch_size, seq_len):
    """Full MoE layer with real weights, A2aSparseMLP swap on (4,8) mesh."""
    xr.set_device_type("TT")
    torch_xla.runtime.use_spmd()

    args = weight_loader.load_config_args()

    # Build the MoE the same way the real model does: Block(layer_id, args).ffn.
    # Full Transformer would need all 43 layers (~520 GB); one Block is the
    # smallest slice that preserves the model's own wiring.
    block = ds_model.Block(layer_id=LAYER_ID, args=args)
    block.ffn.load_state_dict(weight_loader.load_moe_state_dict(layer_id=LAYER_ID))
    block.ffn = block.ffn.eval().to(torch.bfloat16)

    mesh_shape = (2, 4)
    enable_sparse_mlp(block, mesh=mesh_shape, cluster_axis=0, config=args)
    ffn = block.ffn  # now A2aSparseMLPWithSharedExperts

    # Real activations captured from a CPU forward pass through layers
    # 0..n_hash_layers (= layer 3, first score-routed). Replaces torch.randn
    # so the gate sees a realistic distribution and routes to non-uniform
    # experts, which matters for the sparse-expert PCC headroom.
    _, hidden_states = realistic_inputs.get_realistic_inputs(
        layer_id=args.n_hash_layers, batch_size=batch_size, seq_len=seq_len
    )

    num_devices = xr.global_runtime_device_count()
    device_ids = np.array(range(num_devices))
    mesh = Mesh(device_ids, mesh_shape, ("_axis_0", "_axis_1"))

    def get_shard_spec(ffn, args, kwargs):
        shard_specs = {}
        # hidden_states: [batch, seq, dim] — batch on _axis_0, dim on _axis_1
        shard_specs[args[0]] = ("_axis_0", None, None)

        mlp = ffn.mlp if hasattr(ffn, "mlp") else ffn
        shard_specs[mlp.router.gate.weight] = (None, "_axis_1")
        shard_specs[mlp.experts.gate_proj] = (("_axis_0", "_axis_1"), None, None)
        shard_specs[mlp.experts.up_proj] = (("_axis_0", "_axis_1"), None, None)
        shard_specs[mlp.experts.down_proj] = (("_axis_0", "_axis_1"), None, None)

        shared = getattr(ffn, "shared_experts", None)
        if shared is not None:
            shard_specs[shared.w1.weight] = ("_axis_1", None)
            shard_specs[shared.w3.weight] = ("_axis_1", None)
            shard_specs[shared.w2.weight] = (None, "_axis_1")

        return shard_specs

    comparison_config = ComparisonConfig(
        pcc=PccConfig(enabled=True, required_pcc=0.98),
    )

    run_graph_test(
        ffn,
        [hidden_states],
        framework=Framework.TORCH,
        mesh=mesh,
        shard_spec_fn=get_shard_spec,
        comparison_config=comparison_config,
    )


@pytest.mark.nightly
@pytest.mark.llmbox
@pytest.mark.parametrize("batch_size", [32])
@pytest.mark.parametrize("seq_len", [1, 32])
def test_deepseek_v4_flash_moe_hash(batch_size, seq_len):
    """Hash-routed MoE layer (layer_id < n_hash_layers): indices come from a
    learned tid2eid[input_ids] lookup instead of topk over gate scores. The
    rest of the layer (experts, shared expert) is identical to non-hash."""
    xr.set_device_type("TT")
    torch_xla.runtime.use_spmd()

    args = weight_loader.load_config_args()
    assert HASH_LAYER_ID < args.n_hash_layers, (
        f"HASH_LAYER_ID={HASH_LAYER_ID} not a hash layer "
        f"(n_hash_layers={args.n_hash_layers})"
    )

    # Build from Block so the MoE is constructed by the model's own wiring.
    # Hash gate ships tid2eid (int32 [vocab, n_activated_experts]) in place of
    # gate.bias; state_dict loading handles it via gate.tid2eid.
    block = ds_model.Block(layer_id=HASH_LAYER_ID, args=args)
    block.ffn.load_state_dict(weight_loader.load_moe_state_dict(layer_id=HASH_LAYER_ID))
    block.ffn = block.ffn.eval().to(torch.bfloat16)

    mesh_shape = (2, 4)
    enable_sparse_mlp(block, mesh=mesh_shape, cluster_axis=0, config=args)
    runner = _HashMoERunner(block.ffn)

    # Real tokens + real activations from the cached CPU prefix run. For the
    # hash test, `input_ids` matter: gate.tid2eid[input_ids] selects experts,
    # so random ids would give a uniformly-random routing distribution.
    input_ids, hidden_states = realistic_inputs.get_realistic_inputs(
        layer_id=args.n_hash_layers, batch_size=batch_size, seq_len=seq_len
    )

    num_devices = xr.global_runtime_device_count()
    device_ids = np.array(range(num_devices))
    mesh = Mesh(device_ids, mesh_shape, ("_axis_0", "_axis_1"))

    def get_shard_spec(runner, args, kwargs):
        shard_specs = {}
        # hidden_states: [batch, seq, dim]
        shard_specs[args[0]] = ("_axis_0", None, "_axis_1")
        # input_ids: [batch, seq]
        shard_specs[args[1]] = ("_axis_0", None)

        mlp = runner.ffn.mlp
        # Replicate gate.weight instead of sharding on _axis_0. The sharded
        # form decomposes to partial matmul + all_reduce on _axis_0, whose
        # fp32 reduction order differs from CPU's single-shot matmul. Even
        # sub-ULP score differences flip `.topk` choices, so CPU and device
        # select different experts and the residual cascades. See ds4/NOTES.md
        # for the router probe data.
        shard_specs[mlp.router.gate.weight] = (None, "_axis_1")
        # tid2eid [vocab, n_activated_experts] — small, keep replicated.

        shard_specs[mlp.experts.gate_proj] = (("_axis_0", "_axis_1"), None, None)
        shard_specs[mlp.experts.up_proj] = (("_axis_0", "_axis_1"), None, None)
        shard_specs[mlp.experts.down_proj] = (("_axis_0", "_axis_1"), None, None)

        shared = getattr(runner.ffn, "shared_experts", None)
        if shared is not None:
            shard_specs[shared.w1.weight] = ("_axis_1", None)
            shard_specs[shared.w3.weight] = ("_axis_1", None)
            shard_specs[shared.w2.weight] = (None, "_axis_1")

        return shard_specs

    comparison_config = ComparisonConfig(
        pcc=PccConfig(enabled=True, required_pcc=0.99),
    )

    run_graph_test(
        runner,
        [hidden_states, input_ids],
        framework=Framework.TORCH,
        mesh=mesh,
        shard_spec_fn=get_shard_spec,
        comparison_config=comparison_config,
    )
