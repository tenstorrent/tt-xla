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
    combine_shard_specs,
    expert_local_slots,
    make_expert_mapping,
    manual_combine_backward,
    manual_combine_forward,
    reference_combine_forward,
    run_cpu,
    run_tt,
)


class AllToAllCombine(torch.nn.Module):
    def __init__(self, output_shard_dim: int):
        super().__init__()
        self.output_shard_dim = output_shard_dim

    def forward(self, input_tensor, expert_metadata, expert_mapping, expert_locals):
        return torch.ops.tt.all_to_all_combine(
            input_tensor,
            expert_metadata,
            expert_mapping,
            expert_locals,
            num_devices=ROUTING_ROWS,
            cluster_axis=CLUSTER_AXIS,
            num_experts_per_tok=2,
            output_shard_dim=self.output_shard_dim,
        )


def make_mixed_row_expert_metadata(
    bd: int,
    seq: int,
    num_experts_per_tok: int = 2,
    num_experts: int = 8,
) -> torch.Tensor:
    if bd <= 0 or seq <= 0:
        raise ValueError(f"Expected positive bd/seq, got bd={bd}, seq={seq}")
    if num_experts_per_tok <= 0:
        raise ValueError(
            f"Expected positive num_experts_per_tok, got {num_experts_per_tok}"
        )
    if num_experts % ROUTING_ROWS != 0:
        raise ValueError(
            f"num_experts={num_experts} must be divisible by ROUTING_ROWS={ROUTING_ROWS}"
        )

    experts_per_row = num_experts // ROUTING_ROWS
    metadata = torch.empty((1, bd, seq, num_experts_per_tok), dtype=torch.int64)

    # Alternate top-k lanes across mesh rows while cycling columns so the
    # routing pattern spans both rows for every token.
    for bd_idx in range(bd):
        for seq_idx in range(seq):
            base_col = (bd_idx // ROUTING_ROWS + seq_idx) % experts_per_row
            for k_idx in range(num_experts_per_tok):
                row = (bd_idx + k_idx) % ROUTING_ROWS
                col = (base_col + k_idx // ROUTING_ROWS) % experts_per_row
                metadata[0, bd_idx, seq_idx, k_idx] = row * experts_per_row + col

    return metadata


@pytest.mark.nightly
@pytest.mark.parametrize("output_shard_dim", [1, 2])
@pytest.mark.xfail(
    reason="TT all_to_all_combine forward misroutes the second top-k lane when expert metadata spans both mesh rows.",
    strict=True,
)
def test_all_to_all_combine_forward_matches_cpu_and_manual(output_shard_dim: int):
    expert_mapping = make_expert_mapping()

    if output_shard_dim == 1:
        input_tensor = torch.arange(1, 8 * 4 * 2 * 4 + 1, dtype=torch.bfloat16).view(
            8, 4, 2, 4
        )
    else:
        input_tensor = torch.arange(1, 8 * 1 * 4 * 4 + 1, dtype=torch.bfloat16).view(
            8, 1, 4, 4
        )

    bd = input_tensor.shape[1] if output_shard_dim == 1 else input_tensor.shape[2]
    seq = input_tensor.shape[2] if output_shard_dim == 1 else input_tensor.shape[1]
    expert_metadata = make_mixed_row_expert_metadata(bd, seq)
    expert_locals = expert_local_slots(expert_mapping)

    cpu_output, _ = run_cpu(
        AllToAllCombine(output_shard_dim),
        [input_tensor, expert_metadata, expert_mapping, expert_locals],
    )
    tt_output, _ = run_tt(
        AllToAllCombine(output_shard_dim),
        [input_tensor, expert_metadata, expert_mapping, expert_locals],
        shard_specs=combine_shard_specs(output_shard_dim),
    )

    expected_output = manual_combine_forward(
        input_tensor,
        expert_metadata,
        expert_mapping,
        num_devices=ROUTING_ROWS,
        num_experts_per_tok=2,
        output_shard_dim=output_shard_dim,
    )
    torch.testing.assert_close(tt_output, cpu_output, rtol=0, atol=0)
    torch.testing.assert_close(tt_output, expected_output, rtol=0, atol=0)


@pytest.mark.nightly
@pytest.mark.parametrize("output_shard_dim", (1,))
def test_all_to_all_combine_backward_matches_cpu_and_manual_with_cross_row_routing(
    output_shard_dim: int,
):
    H = 7
    K = 2
    S = 2
    BD = 8
    B = BD // ROUTING_ROWS
    E = 2
    expert_mapping = make_expert_mapping()
    expert_metadata = make_mixed_row_expert_metadata(BD, S, K, E)
    expert_locals = expert_local_slots(expert_mapping, num_experts=E)
    print(f"{expert_metadata=}")
    print(f"{expert_locals=}")

    input_tensor = torch.arange(1, E * BD * S * H + 1, dtype=torch.bfloat16).view(
        E, BD, S, H
    )
    gradient = torch.arange(1, K * B * S * H + 1, dtype=torch.bfloat16).view(K, B, S, H)

    input_tensor.requires_grad_(True)

    _, tt_grads = run_tt(
        AllToAllCombine(output_shard_dim),
        [
            input_tensor.clone(),
            expert_metadata.clone(),
            expert_mapping.clone(),
            expert_locals.clone(),
        ],
        shard_specs=combine_shard_specs(output_shard_dim, num_experts=E),
        gradient=gradient.clone(),
    )

    _, cpu_grads = run_cpu(
        AllToAllCombine(output_shard_dim),
        [
            input_tensor.clone(),
            expert_metadata.clone(),
            expert_mapping.clone(),
            expert_locals.clone(),
        ],
        gradient=gradient.clone(),
    )

    expected_grad = manual_combine_backward(
        gradient,
        expert_metadata,
        expert_mapping,
        input_shape=tuple(input_tensor.shape),
        output_shard_dim=output_shard_dim,
    )

    print(f"{tt_grads[0]=}")
    print(f"{cpu_grads[0]=}")
    print(f"{expected_grad=}")

    torch.testing.assert_close(cpu_grads[0], expected_grad, rtol=0, atol=0)
    torch.testing.assert_close(tt_grads[0], expected_grad, rtol=0, atol=0)


@pytest.mark.nightly
@pytest.mark.parametrize("output_shard_dim", [1, 2])
def test_all_to_all_combine_backward_full_mesh_all_experts(
    output_shard_dim: int,
):
    """True all-to-all backward with E=8 experts on a 2x4 mesh.

    Every device owns exactly one expert.  Hand-crafted metadata
    guarantees that (a) all 8 experts appear in the first B=4 batch
    positions used by the backward, and (b) every token's two top-k
    lanes route to experts on *different* mesh rows, so every device
    both sends and receives gradients across the row boundary.

    Expert-to-row mapping (build_expert_mapping with 2x4 mesh):
        even experts (0,2,4,6) → row 0
        odd  experts (1,3,5,7) → row 1
    """
    E = 8
    K = 2
    S = 2
    H = 4
    BD = 8
    B = BD // ROUTING_ROWS

    expert_mapping = make_expert_mapping()
    expert_locals = expert_local_slots(expert_mapping)

    # Explicit routing: each (b, s) pair sends k=0 to one row and k=1
    # to the other.  Across b=0..3 × s=0..1 all 8 experts are hit.
    #   b=0  s=0: [0,1]  s=1: [2,3]
    #   b=1  s=0: [4,5]  s=1: [6,7]
    #   b=2  s=0: [1,0]  s=1: [3,2]
    #   b=3  s=0: [5,4]  s=1: [7,6]
    # BD positions 4..7 are padding (not read by backward).
    expert_metadata = torch.tensor(
        [
            [
                [[0, 1], [2, 3]],
                [[4, 5], [6, 7]],
                [[1, 0], [3, 2]],
                [[5, 4], [7, 6]],
                [[2, 3], [0, 1]],
                [[6, 7], [4, 5]],
                [[3, 2], [1, 0]],
                [[7, 6], [5, 4]],
            ]
        ],
        dtype=torch.int64,
    )

    if output_shard_dim == 1:
        input_tensor = torch.arange(1, E * BD * S * H + 1, dtype=torch.bfloat16).view(
            E, BD, S, H
        )
        gradient = torch.arange(1, K * B * S * H + 1, dtype=torch.bfloat16).view(
            K, B, S, H
        )
    else:
        input_tensor = torch.arange(1, E * S * BD * H + 1, dtype=torch.bfloat16).view(
            E, S, BD, H
        )
        gradient = torch.arange(1, K * S * B * H + 1, dtype=torch.bfloat16).view(
            K, S, B, H
        )

    input_tensor.requires_grad_(True)

    _, tt_grads = run_tt(
        AllToAllCombine(output_shard_dim),
        [
            input_tensor.clone(),
            expert_metadata.clone(),
            expert_mapping.clone(),
            expert_locals.clone(),
        ],
        shard_specs=combine_shard_specs(output_shard_dim),
        gradient=gradient.clone(),
    )

    _, cpu_grads = run_cpu(
        AllToAllCombine(output_shard_dim),
        [
            input_tensor.clone(),
            expert_metadata.clone(),
            expert_mapping.clone(),
            expert_locals.clone(),
        ],
        gradient=gradient.clone(),
    )

    expected_grad = manual_combine_backward(
        gradient,
        expert_metadata,
        expert_mapping,
        input_shape=tuple(input_tensor.shape),
        output_shard_dim=output_shard_dim,
    )

    print(f"{expert_metadata[0, :B]=}")
    print(f"{tt_grads[0]=}")
    print(f"{cpu_grads[0]=}")
    print(f"{expected_grad=}")

    torch.testing.assert_close(cpu_grads[0], expected_grad, rtol=0, atol=0)
    torch.testing.assert_close(tt_grads[0], expected_grad, rtol=0, atol=0)
