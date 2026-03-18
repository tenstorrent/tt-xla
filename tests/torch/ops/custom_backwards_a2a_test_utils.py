#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import copy

import numpy as np
import pytest
import torch
import torch_xla
import torch_xla.distributed.spmd as xs
import torch_xla.runtime as xr
import tt_torch  # noqa: F401
from infra.utilities.torch_multichip_utils import enable_spmd
from torch_xla.distributed.spmd import Mesh
from tt_torch.sparse_mlp import build_expert_mapping

from tests.infra.testers.single_chip.model.torch_model_tester import (
    _mask_jax_accelerator,
)

MESH_SHAPE = (2, 4)
MESH_AXIS_NAMES = ("row", "col")
ROUTING_ROWS = MESH_SHAPE[0]
MESH_COLS = MESH_SHAPE[1]
TOTAL_DEVICES = MESH_SHAPE[0] * MESH_SHAPE[1]
CLUSTER_AXIS = 0


def expert_row_and_local_slot(
    expert_id: int, expert_mapping: torch.Tensor
) -> tuple[int, int]:
    owning_device = int(torch.argmax(expert_mapping[0, 0, expert_id]).item())
    return owning_device // MESH_COLS, owning_device % MESH_COLS


def expert_rows_and_local_slots(
    expert_ids: torch.Tensor, expert_mapping: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    expert_to_device = torch.argmax(expert_mapping[0, 0], dim=-1)
    owning_devices = expert_to_device[expert_ids.to(torch.int64)]
    return owning_devices // MESH_COLS, owning_devices % MESH_COLS


def expert_local_slots(
    expert_mapping: torch.Tensor, num_experts: int | None = None
) -> torch.Tensor:
    """Return the routing-group-local slot for each global expert.

    For a 2x4 mesh with cluster_axis=0 (routing along rows), each row
    is a routing group of MESH_COLS devices.  The local slot of an expert
    is simply the column of its owning device within the row:
        local_slot = owning_device % MESH_COLS

    When num_experts < total experts in expert_mapping, only the first
    num_experts entries are returned so the tensor length matches the
    model's actual expert count E.
    """
    expert_to_device = torch.argmax(expert_mapping[0, 0].to(torch.int64), dim=-1)
    local_slots = expert_to_device % MESH_COLS
    if num_experts is not None:
        local_slots = local_slots[:num_experts]
    return local_slots


def require_2x4_mesh():
    if xr.global_runtime_device_count() != np.prod(MESH_SHAPE):
        pytest.skip(
            f"These tests expect a {MESH_SHAPE} mesh, got "
            f"{xr.global_runtime_device_count()} devices."
        )


def make_mesh():
    require_2x4_mesh()
    device_ids = np.arange(np.prod(MESH_SHAPE))
    return Mesh(device_ids, MESH_SHAPE, MESH_AXIS_NAMES)


def clone_tensor(tensor: torch.Tensor, device=None) -> torch.Tensor:
    cloned = tensor.detach().clone()
    if device is not None:
        cloned = cloned.to(device)
    if tensor.requires_grad:
        cloned.requires_grad_(True)
    return cloned


def clone_inputs(inputs, device=None):
    cloned_inputs = []
    for value in inputs:
        if isinstance(value, torch.Tensor):
            cloned_inputs.append(clone_tensor(value, device=device))
        else:
            cloned_inputs.append(value)
    return cloned_inputs


def to_cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, tuple):
        return tuple(to_cpu(item) for item in value)
    if isinstance(value, list):
        return [to_cpu(item) for item in value]
    return value


def to_device(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, tuple):
        return tuple(to_device(item, device) for item in value)
    if isinstance(value, list):
        return [to_device(item, device) for item in value]
    return value


def collect_input_grads(inputs):
    grads = []
    for value in inputs:
        if isinstance(value, torch.Tensor) and value.requires_grad:
            grads.append(
                None if value.grad is None else value.grad.detach().cpu().clone()
            )
        else:
            grads.append(None)
    return grads


def run_cpu(module: torch.nn.Module, inputs, gradient=None):
    cpu_module = copy.deepcopy(module).train()
    cpu_inputs = clone_inputs(inputs)

    with _mask_jax_accelerator():
        output = cpu_module(*cpu_inputs)
        if gradient is not None:
            if isinstance(output, tuple):
                torch.autograd.backward(output, grad_tensors=gradient)
            else:
                output.backward(gradient=gradient)

    return to_cpu(output), collect_input_grads(cpu_inputs)


def run_fwd_and_bwd_on_tt(module: torch.nn.Module, inputs, gradient):
    output = module(*inputs)
    grad_input = torch.autograd.grad(output, inputs[0], grad_outputs=gradient)[0]
    return output, grad_input


def run_tt(module: torch.nn.Module, inputs, shard_specs, gradient=None):
    xr.set_device_type("TT")
    enable_spmd()
    mesh = make_mesh()
    device = torch_xla.device()

    tt_module = module.to(device).train()
    tt_inputs = clone_inputs(inputs, device=device)
    for input_index, shard_spec in shard_specs.items():
        xs.mark_sharding(tt_inputs[input_index], mesh, shard_spec)

    compile_options = {
        "tt_enable_torch_fx_fusion_pass": False,
        "tt_legacy_compile": True,
    }

    tt_gradient = to_device(gradient, device) if gradient is not None else None
    compiled_fn = torch.compile(
        run_fwd_and_bwd_on_tt, backend="tt", options=compile_options
    )
    output, grad_input = compiled_fn(tt_module, tt_inputs, tt_gradient)
    torch_xla.sync(wait=True)

    return to_cpu(output), {0: to_cpu(grad_input)}


def dispatch_shard_specs():
    return {
        0: ("row", None, None, None),
        1: ("row", None, None, None),
        2: (None, None, None, None),
    }


def combine_shard_specs(output_shard_dim: int, num_experts: int = TOTAL_DEVICES):
    if num_experts % TOTAL_DEVICES == 0:
        dim0_shard = ("col", "row")
    elif num_experts % MESH_COLS == 0:
        dim0_shard = "col"
    elif num_experts % ROUTING_ROWS == 0:
        dim0_shard = "row"
    else:
        dim0_shard = None
    return {
        0: (dim0_shard, None, None, None),
        1: (None, None, None, None),
        2: (None, None, None, None),
        3: (None,),
    }


def remap_shard_specs():
    return {
        0: (None, None, None, ("row", "col")),
        1: (None, None, None, None),
        2: (None, "row", None, None),
    }


def routing_mask(
    expert_indices: torch.Tensor, expert_mapping: torch.Tensor
) -> torch.Tensor:
    batch, _, seq, num_experts_per_tok = expert_indices.shape
    goes_to_device = torch.zeros(batch, seq, ROUTING_ROWS, dtype=torch.bool)

    for b in range(batch):
        for s in range(seq):
            for k in range(num_experts_per_tok):
                expert_id = expert_indices[b, 0, s, k].item()
                owning_row, _ = expert_row_and_local_slot(expert_id, expert_mapping)
                goes_to_device[b, s, owning_row] = True

    return goes_to_device


def manual_dispatch_forward(
    input_tensor: torch.Tensor,
    expert_indices: torch.Tensor,
    expert_mapping: torch.Tensor,
    num_devices: int,
    sparse: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch, _, seq, hidden = input_tensor.shape
    metadata = (
        expert_indices.permute(1, 0, 2, 3).repeat(1, num_devices, 1, 1).contiguous()
    )
    dispatched = torch.zeros(
        1, batch * num_devices, seq, hidden, dtype=input_tensor.dtype
    )
    mask = routing_mask(expert_indices, expert_mapping)

    for b in range(batch):
        for row in range(num_devices):
            bd_index = row * batch + b
            for s in range(seq):
                if sparse and not mask[b, s, row]:
                    continue
                dispatched[0, bd_index, s] = input_tensor[b, 0, s]

    return dispatched, metadata


def manual_dispatch_backward(
    grad_dispatched: torch.Tensor,
    expert_indices: torch.Tensor,
    expert_mapping: torch.Tensor,
) -> torch.Tensor:
    _, bd, seq, hidden = grad_dispatched.shape
    batch = expert_indices.shape[0]
    num_rows = bd // batch
    expected = torch.zeros(1, batch, seq, hidden, dtype=grad_dispatched.dtype)
    grad_by_row = grad_dispatched.view(1, num_rows, batch, seq, hidden)
    mask = routing_mask(expert_indices, expert_mapping)

    for row in range(num_rows):
        row_mask = mask[:, :, row].to(grad_dispatched.dtype).unsqueeze(0).unsqueeze(-1)
        expected += grad_by_row[:, row] * row_mask

    return expected.permute(1, 0, 2, 3).contiguous()


def manual_combine_forward(
    input_tensor: torch.Tensor,
    expert_metadata: torch.Tensor,
    expert_mapping: torch.Tensor,
    num_devices: int,
    num_experts_per_tok: int,
    output_shard_dim: int,
) -> torch.Tensor:
    if output_shard_dim == 1:
        _, bd, seq, hidden = input_tensor.shape
        batch = bd // num_devices
        output = torch.zeros(
            num_experts_per_tok, batch, seq, hidden, dtype=input_tensor.dtype
        )
        for b in range(batch):
            for s in range(seq):
                for k in range(num_experts_per_tok):
                    expert_id = expert_metadata[0, b, s, k].item()
                    row, local_slot = expert_row_and_local_slot(
                        expert_id, expert_mapping
                    )
                    bd_index = row * batch + b
                    output[k, b, s] = input_tensor[local_slot, bd_index, s]
        return output

    _, seq, bd, hidden = input_tensor.shape
    batch = bd // num_devices
    output = torch.zeros(
        num_experts_per_tok, seq, batch, hidden, dtype=input_tensor.dtype
    )
    for b in range(batch):
        for s in range(seq):
            for k in range(num_experts_per_tok):
                expert_id = expert_metadata[0, b, s, k].item()
                row, local_slot = expert_row_and_local_slot(expert_id, expert_mapping)
                bd_index = row * batch + b
                output[k, s, b] = input_tensor[local_slot, s, bd_index]
    return output


def reference_combine_forward(
    input_tensor: torch.Tensor,
    expert_metadata: torch.Tensor,
    expert_mapping: torch.Tensor,
    num_devices: int,
    num_experts_per_tok: int,
    output_shard_dim: int,
) -> torch.Tensor:
    del num_experts_per_tok

    if output_shard_dim == 1:
        _, bd, seq, _ = input_tensor.shape
    else:
        _, seq, bd, _ = input_tensor.shape

    batch = bd // num_devices
    expert_ids = expert_metadata[0, :batch].to(torch.int64)
    rows, local_slots = expert_rows_and_local_slots(expert_ids, expert_mapping)
    batch_indices = torch.arange(
        batch, device=input_tensor.device, dtype=torch.int64
    ).view(batch, 1, 1)
    seq_indices = torch.arange(seq, device=input_tensor.device, dtype=torch.int64).view(
        1, seq, 1
    )
    bd_indices = rows * batch + batch_indices

    if output_shard_dim == 1:
        gathered = input_tensor[local_slots, bd_indices, seq_indices]
        return gathered.permute(2, 0, 1, 3).contiguous()

    gathered = input_tensor[local_slots, seq_indices, bd_indices]
    return gathered.permute(2, 1, 0, 3).contiguous()


def manual_moe_expert_token_remap_forward(
    topk_tensor: torch.Tensor,
    expert_mapping: torch.Tensor,
    expert_metadata: torch.Tensor,
    reduction_size: int,
) -> tuple[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
    import math

    dispatch_groups, batch, seq, num_experts = topk_tensor.shape
    num_experts_per_tok = expert_metadata.shape[-1]
    num_local_experts = num_experts // expert_mapping.shape[-1]
    reduced_seq = math.ceil(batch * seq / reduction_size)
    global_mapping = torch.zeros(1, batch, seq, num_experts, dtype=topk_tensor.dtype)
    global_reduced = torch.zeros(
        1, 1, reduced_seq, num_experts, dtype=topk_tensor.dtype
    )
    local_mapping = torch.zeros(
        1, batch, seq, num_local_experts, dtype=topk_tensor.dtype
    )
    local_reduced = torch.zeros(
        1, 1, reduced_seq, num_local_experts, dtype=topk_tensor.dtype
    )
    local_slots = expert_local_slots(expert_mapping)

    for d in range(dispatch_groups):
        for b in range(batch):
            for s in range(seq):
                chunk_idx = (b * seq + s) // reduction_size
                for k in range(num_experts_per_tok):
                    global_expert = expert_metadata[d, b, s, k].item()
                    local_slot = int(local_slots[global_expert].item())
                    value = topk_tensor[d, b, s, global_expert]
                    global_mapping[0, b, s, global_expert] = value
                    global_reduced[0, 0, chunk_idx, global_expert] = 1.0
                    local_mapping[0, b, s, local_slot] = value
                    local_reduced[0, 0, chunk_idx, local_slot] = 1.0

    return (global_mapping, global_reduced), (local_mapping, local_reduced)


def expand_local_remap_outputs(
    local_mapping: torch.Tensor,
    local_reduced: torch.Tensor,
    expert_mapping: torch.Tensor,
    expert_metadata: torch.Tensor,
    num_experts: int,
    reduction_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    import math

    _, batch, seq, _ = local_mapping.shape
    reduced_seq = math.ceil(batch * seq / reduction_size)
    global_mapping = torch.zeros(
        1,
        batch,
        seq,
        num_experts,
        dtype=local_mapping.dtype,
        device=local_mapping.device,
    )
    global_reduced = torch.zeros(
        1,
        1,
        reduced_seq,
        num_experts,
        dtype=local_reduced.dtype,
        device=local_reduced.device,
    )
    local_slots = expert_local_slots(expert_mapping).to(expert_metadata.device)

    for b in range(batch):
        for s in range(seq):
            chunk_idx = (b * seq + s) // reduction_size
            for global_expert in expert_metadata[0, b, s].tolist():
                local_slot = int(local_slots[global_expert].item())
                global_mapping[0, b, s, global_expert] = local_mapping[
                    0, b, s, local_slot
                ]
                global_reduced[0, 0, chunk_idx, global_expert] = local_reduced[
                    0, 0, chunk_idx, local_slot
                ]

    return global_mapping, global_reduced


def manual_moe_expert_token_remap_backward(
    grad_mapping: torch.Tensor | None,
    grad_reduced: torch.Tensor | None,
    expert_mapping: torch.Tensor,
    expert_metadata: torch.Tensor,
    topk_shape: tuple[int, ...],
    topk_dtype: torch.dtype,
) -> torch.Tensor:
    grad_device = None
    if grad_mapping is not None:
        grad_device = grad_mapping.device
    elif grad_reduced is not None:
        grad_device = grad_reduced.device

    grad_topk = torch.zeros(topk_shape, dtype=topk_dtype, device=grad_device)
    if grad_mapping is not None:
        if grad_mapping.shape[-1] == topk_shape[-1]:
            gathered_grad = torch.gather(grad_mapping, dim=-1, index=expert_metadata)
        else:
            local_slots = expert_local_slots(expert_mapping).to(expert_metadata.device)
            gather_index = local_slots[expert_metadata.to(torch.int64)]
            gathered_grad = torch.gather(grad_mapping, dim=-1, index=gather_index)
        grad_topk.scatter_(dim=-1, index=expert_metadata, src=gathered_grad)

    return grad_topk


def manual_combine_backward(
    grad_output: torch.Tensor,
    expert_metadata: torch.Tensor,
    expert_mapping: torch.Tensor,
    input_shape: tuple[int, ...],
    output_shard_dim: int,
) -> torch.Tensor:
    num_experts = input_shape[0]
    batch = grad_output.shape[1] if output_shard_dim == 1 else grad_output.shape[2]
    seq = grad_output.shape[2] if output_shard_dim == 1 else grad_output.shape[1]
    num_experts_per_tok = grad_output.shape[0]
    expected = torch.zeros(input_shape, dtype=grad_output.dtype)

    for b in range(batch):
        for s in range(seq):
            for k in range(num_experts_per_tok):
                expert_id = expert_metadata[0, b, s, k].item()
                row, _local_slot = expert_row_and_local_slot(expert_id, expert_mapping)
                bd_index = row * batch + b
                if output_shard_dim == 1:
                    expected[expert_id, bd_index, s] = grad_output[k, b, s]
                else:
                    expected[expert_id, s, bd_index] = grad_output[k, s, b]

    return expected


def make_expert_mapping(num_experts: int = TOTAL_DEVICES) -> torch.Tensor:
    mesh_shape = MESH_SHAPE if num_experts == TOTAL_DEVICES else None
    expert_mapping = build_expert_mapping(
        num_experts=num_experts,
        num_devices=TOTAL_DEVICES,
        mesh_shape=mesh_shape,
    )
    return expert_mapping
