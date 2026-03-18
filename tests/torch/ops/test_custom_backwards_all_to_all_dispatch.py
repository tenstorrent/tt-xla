#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from custom_backwards_a2a_test_utils import (
    CLUSTER_AXIS,
    ROUTING_ROWS,
    dispatch_shard_specs,
    make_expert_mapping,
    manual_dispatch_backward,
    manual_dispatch_forward,
    run_cpu,
    run_tt,
)


class AllToAllDispatchForward(torch.nn.Module):
    def forward(self, input_tensor, expert_indices, expert_mapping):
        return torch.ops.tt.all_to_all_dispatch(
            input_tensor,
            expert_indices,
            expert_mapping,
            num_devices=ROUTING_ROWS,
            cluster_axis=CLUSTER_AXIS,
        )


class AllToAllDispatchBackward(torch.nn.Module):
    def forward(self, input_tensor, expert_indices, expert_mapping):
        dispatched, _ = torch.ops.tt.all_to_all_dispatch(
            input_tensor,
            expert_indices,
            expert_mapping,
            num_devices=ROUTING_ROWS,
            cluster_axis=CLUSTER_AXIS,
        )
        return dispatched


@pytest.mark.nightly
def test_all_to_all_dispatch_forward_cpu_matches_manual_replication():
    input_tensor = torch.arange(1, 2 * 1 * 2 * 4 + 1, dtype=torch.bfloat16).view(
        2, 1, 2, 4
    )
    expert_indices = torch.tensor(
        [[[[0, 1], [2, 3]]], [[[4, 5], [6, 7]]]],
        dtype=torch.int64,
    )
    expert_mapping = make_expert_mapping()

    cpu_output, _ = run_cpu(
        AllToAllDispatchForward(),
        [input_tensor, expert_indices, expert_mapping],
    )

    cpu_dispatched, cpu_metadata = cpu_output
    expected_cpu_dispatched, expected_cpu_metadata = manual_dispatch_forward(
        input_tensor,
        expert_indices,
        expert_mapping,
        num_devices=ROUTING_ROWS,
        sparse=False,
    )
    torch.testing.assert_close(cpu_metadata, expected_cpu_metadata, rtol=0, atol=0)
    torch.testing.assert_close(cpu_dispatched, expected_cpu_dispatched, rtol=0, atol=0)


@pytest.mark.nightly
def test_all_to_all_dispatch_forward_tt_matches_manual_sparse_routing():
    input_tensor = torch.arange(1, 2 * 1 * 2 * 4 + 1, dtype=torch.bfloat16).view(
        2, 1, 2, 4
    )
    expert_indices = torch.tensor(
        [[[[0, 1], [2, 3]]], [[[4, 5], [6, 7]]]],
        dtype=torch.int64,
    )
    expert_mapping = make_expert_mapping()

    tt_output, _ = run_tt(
        AllToAllDispatchForward(),
        [input_tensor, expert_indices, expert_mapping],
        shard_specs=dispatch_shard_specs(),
    )

    tt_dispatched, tt_metadata = tt_output
    expected_tt_dispatched, expected_tt_metadata = manual_dispatch_forward(
        input_tensor,
        expert_indices,
        expert_mapping,
        num_devices=ROUTING_ROWS,
        sparse=True,
    )
    torch.testing.assert_close(tt_metadata, expected_tt_metadata, rtol=0, atol=0)
    torch.testing.assert_close(tt_dispatched, expected_tt_dispatched, rtol=0, atol=0)


@pytest.mark.nightly
def test_all_to_all_dispatch_backward_cpu_matches_manual_routing():
    input_tensor = torch.arange(1, 2 * 1 * 2 * 4 + 1, dtype=torch.bfloat16).view(
        2, 1, 2, 4
    )
    input_tensor.requires_grad_(True)
    expert_indices = torch.tensor(
        [[[[0, 2], [1, 3]]], [[[4, 5], [6, 7]]]],
        dtype=torch.int64,
    )
    expert_mapping = make_expert_mapping()
    gradient = torch.arange(1, 1 * 4 * 2 * 4 + 1, dtype=torch.bfloat16).view(1, 4, 2, 4)

    _, cpu_grads = run_cpu(
        AllToAllDispatchBackward(),
        [input_tensor, expert_indices, expert_mapping],
        gradient=gradient,
    )
    expected_grad = manual_dispatch_backward(gradient, expert_indices, expert_mapping)
    torch.testing.assert_close(cpu_grads[0], expected_grad, rtol=0, atol=0)


@pytest.mark.nightly
def test_all_to_all_dispatch_backward_tt_matches_manual_routing():
    input_tensor = torch.arange(1, 2 * 1 * 2 * 4 + 1, dtype=torch.bfloat16).view(
        2, 1, 2, 4
    )
    input_tensor.requires_grad_(True)
    expert_indices = torch.tensor(
        [[[[0, 2], [1, 3]]], [[[4, 5], [6, 7]]]],
        dtype=torch.int64,
    )
    expert_mapping = make_expert_mapping()
    gradient = torch.arange(1, 1 * 4 * 2 * 4 + 1, dtype=torch.bfloat16).view(1, 4, 2, 4)

    _, tt_grads = run_tt(
        AllToAllDispatchBackward(),
        [input_tensor, expert_indices, expert_mapping],
        shard_specs=dispatch_shard_specs(),
        gradient=gradient,
    )

    expected_grad = manual_dispatch_backward(gradient, expert_indices, expert_mapping)
    torch.testing.assert_close(tt_grads[0], expected_grad, rtol=0, atol=0)
