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
    expand_local_remap_outputs,
    make_expert_mapping,
    manual_moe_expert_token_remap_backward,
    manual_moe_expert_token_remap_forward,
    remap_shard_specs,
    run_cpu,
    run_tt,
)

REDUCTION_SIZE = 2
NUM_EXPERTS = 32
NUM_LOCAL_EXPERTS = NUM_EXPERTS // 8


class MoeExpertTokenRemap(torch.nn.Module):
    def forward(self, topk_tensor, expert_mapping, expert_metadata):
        return torch.ops.tt.moe_expert_token_remap(
            topk_tensor,
            expert_mapping,
            expert_metadata,
            reduction_size=REDUCTION_SIZE,
        )


def make_remap_inputs():
    topk_tensor = torch.arange(
        1, 1 * 4 * 2 * NUM_EXPERTS + 1, dtype=torch.bfloat16
    ).view(1, 4, 2, NUM_EXPERTS)
    expert_mapping = make_expert_mapping(num_experts=NUM_EXPERTS)
    expert_metadata = torch.tensor(
        [
            [
                [[0, 5], [8, 13]],
                [[16, 21], [24, 29]],
                [[1, 6], [9, 14]],
                [[17, 22], [25, 30]],
            ]
        ],
        dtype=torch.int64,
    )
    return topk_tensor, expert_mapping, expert_metadata


@pytest.mark.nightly
def test_moe_expert_token_remap_forward_cpu_matches_manual():
    topk_tensor, expert_mapping, expert_metadata = make_remap_inputs()

    cpu_output, _ = run_cpu(
        MoeExpertTokenRemap(),
        [topk_tensor, expert_mapping, expert_metadata],
    )
    (expected_mapping, expected_reduced), _ = manual_moe_expert_token_remap_forward(
        topk_tensor,
        expert_mapping,
        expert_metadata,
        reduction_size=REDUCTION_SIZE,
    )

    cpu_mapping, cpu_reduced = cpu_output
    torch.testing.assert_close(cpu_mapping, expected_mapping, rtol=0, atol=0)
    torch.testing.assert_close(cpu_reduced, expected_reduced, rtol=0, atol=0)


@pytest.mark.nightly
def test_moe_expert_token_remap_backward_cpu_matches_manual():
    topk_tensor, expert_mapping, expert_metadata = make_remap_inputs()
    topk_tensor.requires_grad_(True)

    grad_mapping = torch.arange(
        1, 1 * 4 * 2 * NUM_EXPERTS + 1, dtype=torch.bfloat16
    ).view(1, 4, 2, NUM_EXPERTS)
    grad_reduced = torch.zeros(1, 1, 4, NUM_EXPERTS, dtype=torch.bfloat16)

    _, cpu_grads = run_cpu(
        MoeExpertTokenRemap(),
        [topk_tensor, expert_mapping, expert_metadata],
        gradient=(grad_mapping, grad_reduced),
    )
    expected_grad = manual_moe_expert_token_remap_backward(
        grad_mapping,
        grad_reduced,
        expert_mapping,
        expert_metadata,
        topk_shape=tuple(topk_tensor.shape),
        topk_dtype=topk_tensor.dtype,
    )

    torch.testing.assert_close(cpu_grads[0], expected_grad, rtol=0, atol=0)


@pytest.mark.nightly
@pytest.mark.xfail(
    reason="TT moe_expert_token_remap forward still mis-materializes local-E outputs for this isolated op test.",
    strict=True,
)
def test_moe_expert_token_remap_forward_tt_matches_cpu_and_manual():
    topk_tensor, expert_mapping, expert_metadata = make_remap_inputs()

    cpu_output, _ = run_cpu(
        MoeExpertTokenRemap(),
        [topk_tensor, expert_mapping, expert_metadata],
    )
    tt_output, _ = run_tt(
        MoeExpertTokenRemap(),
        [topk_tensor, expert_mapping, expert_metadata],
        shard_specs=remap_shard_specs(),
    )

    (expected_mapping, expected_reduced), (
        expected_local_mapping,
        expected_local_reduced,
    ) = manual_moe_expert_token_remap_forward(
        topk_tensor,
        expert_mapping,
        expert_metadata,
        reduction_size=REDUCTION_SIZE,
    )
    cpu_mapping, cpu_reduced = cpu_output
    tt_mapping, tt_reduced = tt_output
    tt_mapping_expanded, tt_reduced_expanded = expand_local_remap_outputs(
        tt_mapping,
        tt_reduced,
        expert_mapping,
        expert_metadata,
        num_experts=NUM_EXPERTS,
        reduction_size=REDUCTION_SIZE,
    )
    torch.testing.assert_close(tt_mapping, expected_local_mapping, rtol=0, atol=0)
    torch.testing.assert_close(tt_reduced, expected_local_reduced, rtol=0, atol=0)
    torch.testing.assert_close(tt_mapping_expanded, cpu_mapping, rtol=0, atol=0)
    torch.testing.assert_close(tt_reduced_expanded, cpu_reduced, rtol=0, atol=0)
    torch.testing.assert_close(tt_mapping_expanded, expected_mapping, rtol=0, atol=0)
    torch.testing.assert_close(tt_reduced_expanded, expected_reduced, rtol=0, atol=0)


@pytest.mark.nightly
def test_moe_expert_token_remap_backward_tt_matches_cpu_and_manual():
    topk_tensor, expert_mapping, expert_metadata = make_remap_inputs()
    topk_tensor.requires_grad_(True)

    grad_mapping_local = torch.arange(
        1, 1 * 4 * 2 * NUM_LOCAL_EXPERTS + 1, dtype=torch.bfloat16
    ).view(1, 4, 2, NUM_LOCAL_EXPERTS)
    grad_reduced_local = torch.zeros(1, 1, 4, NUM_LOCAL_EXPERTS, dtype=torch.bfloat16)
    grad_mapping, grad_reduced = expand_local_remap_outputs(
        grad_mapping_local,
        grad_reduced_local,
        expert_mapping,
        expert_metadata,
        num_experts=NUM_EXPERTS,
        reduction_size=REDUCTION_SIZE,
    )

    _, cpu_grads = run_cpu(
        MoeExpertTokenRemap(),
        [topk_tensor, expert_mapping, expert_metadata],
        gradient=(grad_mapping, grad_reduced),
    )
    _, tt_grads = run_tt(
        MoeExpertTokenRemap(),
        [topk_tensor, expert_mapping, expert_metadata],
        shard_specs=remap_shard_specs(),
        gradient=(grad_mapping_local, grad_reduced_local),
    )

    expected_grad = manual_moe_expert_token_remap_backward(
        grad_mapping_local,
        grad_reduced_local,
        expert_mapping,
        expert_metadata,
        topk_shape=tuple(topk_tensor.shape),
        topk_dtype=topk_tensor.dtype,
    )
    torch.testing.assert_close(tt_grads[0], cpu_grads[0], rtol=0, atol=0)
    torch.testing.assert_close(tt_grads[0], expected_grad, rtol=0, atol=0)
