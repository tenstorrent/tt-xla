# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import torch
import torch_xla
import torch_xla.distributed.spmd as xs
import torch_xla.runtime as xr
from benchmark.utils import compute_pcc
from infra.utilities.torch_multichip_utils import enable_spmd
from torch_xla.distributed.spmd import Mesh
from transformers import AutoModelForCausalLM
from tt_torch.sparse_mlp import A2aSparseMLP

from third_party.tt_forge_models.gpt_oss.pytorch.loader import (
    ModelLoader as GPTOSSModelLoader,
)
from third_party.tt_forge_models.gpt_oss.pytorch.overrides import (
    override_gpt_oss_modules,
)


def _setup_mesh():
    enable_spmd()
    xr.set_device_type("TT")
    num_devices = xr.global_runtime_device_count()
    mesh_shape = (1, num_devices)
    device_ids = np.array(range(num_devices))
    mesh = Mesh(device_ids, mesh_shape, ("batch", "model"))
    return mesh, mesh_shape, torch_xla.device(), num_devices


def _build_model(num_devices: int):
    config = GPTOSSModelLoader(num_layers=1).load_config()
    config.num_hidden_layers = 1
    config.num_local_experts = 4
    config.num_experts_per_tok = 2
    config.hidden_size = 64
    config.intermediate_size = 64
    assert config.num_local_experts % num_devices == 0

    model = AutoModelForCausalLM.from_config(
        config, trust_remote_code=True, torch_dtype=torch.float32
    )
    override_gpt_oss_modules(model)
    return model, config


def _wrap_with_a2a(mlp, config, num_devices: int) -> A2aSparseMLP:
    a2a = A2aSparseMLP(
        mlp,
        num_experts=config.num_local_experts,
        num_experts_per_tok=config.num_experts_per_tok,
        num_devices=num_devices,
        dispatch_devices=num_devices,
        cluster_axis=1,
        config=config,
        use_dense_matmul=True,
        deinterleave_fused_experts=True,
    )
    return a2a


def _shard_a2a_mlp(mlp: A2aSparseMLP, mesh: Mesh) -> None:
    xs.mark_sharding(mlp.expert_mapping, mesh, (None, None, None, "model"))
    xs.mark_sharding(mlp.experts.gate_proj, mesh, ("model", None, None))
    xs.mark_sharding(mlp.experts.gate_proj_bias, mesh, ("model", None))
    xs.mark_sharding(mlp.experts.up_proj, mesh, ("model", None, None))
    xs.mark_sharding(mlp.experts.up_proj_bias, mesh, ("model", None))
    xs.mark_sharding(mlp.experts.down_proj, mesh, ("model", None, None))
    xs.mark_sharding(mlp.experts.down_proj_bias, mesh, ("model", None))


@pytest.mark.push
@pytest.mark.dual_chip
def test_a2a_sparse_mlp_backward_pcc():
    """A2aSparseMLP forward+backward on N300 (1x2 mesh) vs CPU.

    Uses the de-interleaved GPT-OSS layout (separate gate_proj / up_proj) to
    avoid the strided-slice [..., ::2] backward miscompile on TT XLA.
    """
    mesh, mesh_shape, device, num_devices = _setup_mesh()
    batch_size, seq_len = 2, 32

    torch.manual_seed(0)
    model_cpu, config = _build_model(num_devices=num_devices)
    model_tt, _ = _build_model(num_devices=num_devices)
    model_tt.load_state_dict(model_cpu.state_dict())

    mlp_cpu = _wrap_with_a2a(
        model_cpu.model.layers[0].mlp, config, num_devices=num_devices
    )
    mlp_tt = _wrap_with_a2a(
        model_tt.model.layers[0].mlp, config, num_devices=num_devices
    )

    hidden_states = torch.randn(
        (batch_size, seq_len, config.hidden_size), dtype=torch.float32
    )

    x_cpu = hidden_states.detach().clone().requires_grad_(True)
    out_cpu, _ = mlp_cpu(x_cpu)
    out_cpu.sum().backward()

    mlp_tt = mlp_tt.to(device)
    x_tt = hidden_states.detach().clone().to(device).requires_grad_(True)
    _shard_a2a_mlp(mlp_tt, mesh)

    out_tt, _ = mlp_tt(x_tt)
    out_tt.sum().backward()

    required_pcc = 0.99
    compute_pcc(out_cpu, out_tt.cpu(), required_pcc)
    compute_pcc(x_cpu.grad, x_tt.grad.cpu(), required_pcc)
    compute_pcc(
        mlp_cpu.experts.gate_proj.grad,
        mlp_tt.experts.gate_proj.grad.cpu(),
        required_pcc,
    )
    compute_pcc(
        mlp_cpu.experts.up_proj.grad,
        mlp_tt.experts.up_proj.grad.cpu(),
        required_pcc,
    )
    compute_pcc(
        mlp_cpu.experts.down_proj.grad,
        mlp_tt.experts.down_proj.grad.cpu(),
        required_pcc,
    )
    compute_pcc(
        mlp_cpu.router.weight.grad,
        mlp_tt.router.weight.grad.cpu(),
        required_pcc,
    )
