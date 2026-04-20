# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch_xla
import torch_xla.distributed.spmd as xs
import torch_xla.runtime as xr
from benchmark.utils import compute_pcc
from infra.utilities.torch_multichip_utils import enable_spmd
from torch_xla.distributed.spmd import Mesh
from tt_torch.sparse_mlp import A2aSparseMLP, FusedExpertsWrapper


@dataclass
class _MoEConfig:
    hidden_size: int
    intermediate_size: int
    num_local_experts: int
    num_experts_per_tok: int


class _Router(nn.Module):
    def __init__(self, hidden_size: int, num_experts: int, num_experts_per_tok: int):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(num_experts, hidden_size) * 0.02)
        self.num_experts_per_tok = num_experts_per_tok

    def forward(self, hidden_states: torch.Tensor):
        logits = hidden_states @ self.weight.t()
        scores = torch.softmax(logits, dim=-1)
        _, indices = torch.topk(scores, self.num_experts_per_tok, dim=-1)
        return scores, indices


class _Experts(nn.Module):
    def __init__(self, config: _MoEConfig):
        super().__init__()
        E, H, I = config.num_local_experts, config.hidden_size, config.intermediate_size
        self.intermediate_size = I
        self.gate_up_proj = nn.Parameter(torch.randn(E, H, 2 * I) * 0.02)
        self.gate_up_proj_bias = nn.Parameter(torch.zeros(E, 2 * I))
        self.down_proj = nn.Parameter(torch.randn(E, I, H) * 0.02)
        self.down_proj_bias = nn.Parameter(torch.zeros(E, H))


class _MoEMLP(nn.Module):
    def __init__(self, config: _MoEConfig):
        super().__init__()
        self.config = config
        self.router = _Router(
            config.hidden_size, config.num_local_experts, config.num_experts_per_tok
        )
        self.experts = _Experts(config)

    def forward(self, hidden_states: torch.Tensor):
        B, S, H = hidden_states.shape
        E = self.config.num_local_experts

        flat = hidden_states.reshape(B * S, H)
        scores, indices = self.router(flat)

        gate_up = torch.einsum("nh,eho->neo", flat, self.experts.gate_up_proj)
        gate_up = gate_up + self.experts.gate_up_proj_bias
        gate = gate_up[..., ::2].clamp(max=7.0)
        up = gate_up[..., 1::2].clamp(-7.0, 7.0)
        activated = (up + 1) * gate * torch.sigmoid(gate * 1.702)
        expert_out = torch.einsum("nei,eih->neh", activated, self.experts.down_proj)
        expert_out = expert_out + self.experts.down_proj_bias

        weights = torch.zeros(B * S, E, dtype=hidden_states.dtype)
        weights.scatter_(-1, indices, scores.gather(-1, indices))
        out = (expert_out * weights.unsqueeze(-1)).sum(dim=1)
        return out.reshape(B, S, H), scores


def _setup_mesh():
    enable_spmd()
    xr.set_device_type("TT")
    num_devices = xr.global_runtime_device_count()
    mesh_shape = (1, num_devices)
    device_ids = np.array(range(num_devices))
    mesh = Mesh(device_ids, mesh_shape, ("batch", "model"))
    return mesh, torch_xla.device(), num_devices


def _wrap_with_a2a(mlp: _MoEMLP, config: _MoEConfig, num_devices: int) -> A2aSparseMLP:
    mlp.experts = FusedExpertsWrapper(mlp.experts)
    a2a = A2aSparseMLP(
        mlp,
        num_experts=config.num_local_experts,
        num_experts_per_tok=config.num_experts_per_tok,
        num_devices=num_devices,
        dispatch_devices=num_devices,
        cluster_axis=1,
        config=config,
    )
    a2a.use_dense_matmul = True
    return a2a


def _shard_a2a_mlp(mlp: A2aSparseMLP, mesh: Mesh) -> None:
    experts = mlp.experts
    xs.mark_sharding(mlp.expert_mapping, mesh, (None, None, None, "model"))
    xs.mark_sharding(experts.gate_up_proj, mesh, ("model", None, None))
    xs.mark_sharding(experts.gate_up_proj_bias, mesh, ("model", None))
    xs.mark_sharding(experts.down_proj, mesh, ("model", None, None))
    xs.mark_sharding(experts.down_proj_bias, mesh, ("model", None))


@pytest.mark.push
@pytest.mark.dual_chip
def test_a2a_sparse_mlp_backward_pcc():
    mesh, device, num_devices = _setup_mesh()

    config = _MoEConfig(
        hidden_size=32, intermediate_size=64, num_local_experts=4, num_experts_per_tok=2
    )
    batch_size, seq_len = 1, 32

    torch.manual_seed(0)
    mlp_cpu = _MoEMLP(config)
    hidden_states = torch.randn(batch_size, seq_len, config.hidden_size, dtype=torch.bfloat16)
    mlp_cpu = mlp_cpu.to(torch.bfloat16)

    x_cpu = hidden_states.detach().clone().requires_grad_(True)
    out_cpu, _ = mlp_cpu(x_cpu)
    out_cpu.sum().backward()

    mlp_tt = _MoEMLP(config).to(torch.bfloat16)
    mlp_tt.load_state_dict(mlp_cpu.state_dict())
    mlp_tt = _wrap_with_a2a(mlp_tt, config, num_devices=num_devices)
    mlp_tt = mlp_tt.to(device)

    x_tt = hidden_states.detach().clone().to(device).requires_grad_(True)
    _shard_a2a_mlp(mlp_tt, mesh)

    out_tt, _ = mlp_tt(x_tt)
    out_tt.sum().backward()

    required_pcc = 0.99
    compute_pcc(x_cpu.grad, x_tt.grad.cpu(), required_pcc)
    compute_pcc(
        mlp_cpu.experts.gate_up_proj.grad,
        mlp_tt.experts.gate_up_proj.grad.cpu(),
        required_pcc,
    )
    compute_pcc(
        mlp_cpu.experts.down_proj.grad,
        mlp_tt.experts.down_proj.grad.cpu(),
        required_pcc,
    )
    compute_pcc(
        mlp_cpu.router.weight.grad, mlp_tt.router.weight.grad.cpu(), required_pcc
    )
