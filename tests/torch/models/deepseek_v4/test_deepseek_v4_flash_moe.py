# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import importlib.util
import os
import sys

import numpy as np
import pytest
import torch
import torch_xla
import torch_xla.runtime as xr
from infra import Framework, run_graph_test
from infra.evaluators import ComparisonConfig, PccConfig
from torch_xla.distributed.spmd import Mesh
from tt_torch.sparse_mlp import enable_sparse_mlp


def _register_kernel_stub():
    """Pre-register our bf16 kernel stub as the top-level `kernel` module so
    the upstream model.py's `from kernel import ...` resolves to it instead of
    the tilelang-backed kernel that ships alongside it in tt_forge_models.
    Must run before importing the upstream model."""
    stub_path = os.path.join(os.path.dirname(__file__), "kernel.py")
    spec = importlib.util.spec_from_file_location("kernel", stub_path)
    stub = importlib.util.module_from_spec(spec)
    sys.modules["kernel"] = stub
    spec.loader.exec_module(stub)


_register_kernel_stub()

from third_party.tt_forge_models.deepseek_v4.original_model import (  # noqa: E402
    model as ds_model,
)

from . import weight_loader  # noqa: E402

# Layer past `n_hash_layers` so Gate uses score-based routing (no input_ids
# dependency). Layer 10 is a typical routed-MoE layer in V4-Flash (n_hash=3).
LAYER_ID = 10
# Layer < n_hash_layers uses static hash routing via gate.tid2eid[input_ids].
HASH_LAYER_ID = 0


def _patch_model_for_a2a_compat():
    """Make V4 MoE/Gate callable with input_ids=None so A2aSparseMLP (which
    only passes hidden_states) can dispatch through them.

    For hash layers, input_ids is required by `gate.tid2eid[input_ids]`. The
    hash test injects it via a tensor attribute `gate._ambient_input_ids` set
    on the gate immediately before the a2a forward; the patched Gate/MoE
    forwards below read that attribute when no explicit arg is given."""

    orig_moe = ds_model.MoE.forward

    def moe_forward(self, x, input_ids=None):
        if input_ids is None:
            amb = getattr(self.gate, "_ambient_input_ids", None)
            if amb is not None:
                input_ids = amb
            else:
                # Non-hash layers ignore input_ids; zeros is a safe filler.
                input_ids = torch.zeros(x.shape[:-1], dtype=torch.long, device=x.device)
        return orig_moe(self, x, input_ids)

    ds_model.MoE.forward = moe_forward

    def gate_forward(self, x, input_ids=None):
        # Full re-implementation of Gate.forward that:
        # 1) Pulls input_ids from _ambient_input_ids when not passed (needed for
        #    the a2a RouterAdapter which calls gate(x) with one arg).
        # 2) Replaces `original_scores.gather(1, indices)` with one-hot ×
        #    elementwise × sum. The .gather path lowers to an all_gather
        #    followed by a flat embedding lookup whose per-shard batch offset
        #    is lost under SPMD (the row index stays arange(local_batch) on
        #    every shard), which silently corrupts routing weights on every
        #    non-first batch-axis shard. See ds4/log0.log TTNN dump for the
        #    concrete pattern.
        if input_ids is None and self.hash:
            input_ids = getattr(self, "_ambient_input_ids", None)

        scores = ds_model.linear(x.float(), self.weight.float())
        if self.score_func == "softmax":
            scores = scores.softmax(dim=-1)
        elif self.score_func == "sigmoid":
            scores = scores.sigmoid()
        else:
            scores = torch.nn.functional.softplus(scores).sqrt()
        original_scores = scores
        if self.bias is not None:
            scores = scores + self.bias
        if self.hash:
            indices = self.tid2eid[input_ids]
        else:
            indices = scores.topk(self.topk, dim=-1)[1]

        one_hot = torch.nn.functional.one_hot(
            indices.long(), num_classes=original_scores.size(-1)
        ).to(original_scores.dtype)
        weights = (one_hot * original_scores.unsqueeze(1)).sum(dim=-1)

        if self.score_func != "softmax":
            weights /= weights.sum(dim=-1, keepdim=True)
        weights *= self.route_scale
        return weights, indices

    ds_model.Gate.forward = gate_forward


_patch_model_for_a2a_compat()


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
@pytest.mark.parametrize("batch_size", [32])
@pytest.mark.parametrize("seq_len", [32])
def test_deepseek_v4_flash_mlp(batch_size, seq_len):
    """Single Expert (SwiGLU MLP) with real pretrained weights, sharded on (4,8)."""
    xr.set_device_type("TT")
    torch_xla.runtime.use_spmd()

    args = weight_loader.load_config_args()

    expert = ds_model.Expert(
        args.dim,
        args.moe_inter_dim,
        dtype=None,
        swiglu_limit=args.swiglu_limit,
    )
    expert = expert.eval().to(torch.bfloat16)

    state_dict = weight_loader.load_expert_state_dict(layer_id=LAYER_ID, expert_id=0)
    expert.load_state_dict(state_dict)

    x = torch.randn(batch_size * seq_len, args.dim, dtype=torch.bfloat16)

    mesh_shape = (4, 8)
    num_devices = xr.global_runtime_device_count()
    device_ids = np.array(range(num_devices))
    mesh = Mesh(device_ids, mesh_shape, ("_axis_0", "_axis_1"))

    def get_shard_spec(expert, args, kwargs):
        return {
            args[0]: ("_axis_0", None),  # tokens on _axis_0
            expert.w1.weight: ("_axis_1", None),  # column-parallel
            expert.w3.weight: ("_axis_1", None),  # column-parallel
            expert.w2.weight: (None, "_axis_1"),  # row-parallel
        }

    comparison_config = ComparisonConfig(
        pcc=PccConfig(enabled=True, required_pcc=0.99),
    )

    run_graph_test(
        expert,
        [x],
        framework=Framework.TORCH,
        mesh=mesh,
        shard_spec_fn=get_shard_spec,
        comparison_config=comparison_config,
    )


@pytest.mark.nightly
@pytest.mark.llmbox
@pytest.mark.parametrize("batch_size", [32])
@pytest.mark.parametrize("seq_len", [32])
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

    mesh_shape = (4, 8)
    enable_sparse_mlp(block, mesh=mesh_shape, cluster_axis=0, config=args)
    ffn = block.ffn  # now A2aSparseMLPWithSharedExperts

    hidden_states = torch.randn((batch_size, seq_len, args.dim), dtype=torch.bfloat16)

    num_devices = xr.global_runtime_device_count()
    device_ids = np.array(range(num_devices))
    mesh = Mesh(device_ids, mesh_shape, ("_axis_0", "_axis_1"))

    def get_shard_spec(ffn, args, kwargs):
        shard_specs = {}
        # hidden_states: [batch, seq, dim] — batch on _axis_0, dim on _axis_1
        shard_specs[args[0]] = ("_axis_0", None, None)

        mlp = ffn.mlp if hasattr(ffn, "mlp") else ffn
        # Replicate gate.weight instead of sharding on _axis_0. The sharded
        # form decomposes to partial matmul + all_reduce on _axis_0, whose
        # fp32 reduction order differs from CPU's single-shot matmul. Even
        # sub-ULP score differences flip `.topk` choices, so CPU and device
        # select different experts and the residual cascades. See ds4/NOTES.md
        # for the router probe data.
        shard_specs[mlp.router.gate.weight] = (None, None)
        shard_specs[mlp.experts.gate_proj] = (("_axis_0", "_axis_1"), None, None)
        shard_specs[mlp.experts.up_proj] = (("_axis_0", "_axis_1"), None, None)
        shard_specs[mlp.experts.down_proj] = (("_axis_0", "_axis_1"), None, None)

        shared = getattr(ffn, "shared_experts", None)
        if shared is not None:
            shard_specs[shared.w1.weight] = (None, None)
            shard_specs[shared.w3.weight] = (None, None)
            shard_specs[shared.w2.weight] = (None, None)

        return shard_specs

    comparison_config = ComparisonConfig(
        pcc=PccConfig(enabled=True, required_pcc=0.99),
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
@pytest.mark.parametrize("seq_len", [32])
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

    mesh_shape = (4, 8)
    enable_sparse_mlp(block, mesh=mesh_shape, cluster_axis=0, config=args)
    runner = _HashMoERunner(block.ffn)

    hidden_states = torch.randn((batch_size, seq_len, args.dim), dtype=torch.bfloat16)
    input_ids = torch.randint(
        0, args.vocab_size, (batch_size, seq_len), dtype=torch.long
    )

    num_devices = xr.global_runtime_device_count()
    device_ids = np.array(range(num_devices))
    mesh = Mesh(device_ids, mesh_shape, ("_axis_0", "_axis_1"))

    def get_shard_spec(runner, args, kwargs):
        shard_specs = {}
        # hidden_states: [batch, seq, dim]
        shard_specs[args[0]] = ("_axis_1", None, "_axis_0")
        # input_ids: [batch, seq]
        shard_specs[args[1]] = ("_axis_1", None)

        mlp = runner.ffn.mlp
        # Replicate gate.weight instead of sharding on _axis_0. The sharded
        # form decomposes to partial matmul + all_reduce on _axis_0, whose
        # fp32 reduction order differs from CPU's single-shot matmul. Even
        # sub-ULP score differences flip `.topk` choices, so CPU and device
        # select different experts and the residual cascades. See ds4/NOTES.md
        # for the router probe data.
        shard_specs[mlp.router.gate.weight] = (None, None)
        # tid2eid [vocab, n_activated_experts] — small, keep replicated.

        shard_specs[mlp.experts.gate_proj] = (("_axis_0", "_axis_1"), None, None)
        shard_specs[mlp.experts.up_proj] = (("_axis_0", "_axis_1"), None, None)
        shard_specs[mlp.experts.down_proj] = (("_axis_0", "_axis_1"), None, None)

        shared = getattr(runner.ffn, "shared_experts", None)
        if shared is not None:
            shard_specs[shared.w1.weight] = (None, None)
            shard_specs[shared.w3.weight] = (None, None)
            shard_specs[shared.w2.weight] = (None, None)

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
