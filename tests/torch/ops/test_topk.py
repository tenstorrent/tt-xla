# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from benchmark.utils import compute_pcc
from infra import Framework, run_op_test_with_random_inputs
from utils import Category


def topk_indices_comparator(device_output, golden_output, args, kwargs):
    """Comparator for topk operation returning indices only."""
    device_indices = device_output
    golden_indices = golden_output
    input_tensor = args[0]

    # Move device outputs from XLA to CPU for comparison
    device_indices = device_indices.cpu()

    # 1) Assert device_indices has no duplicate elements (per row, along last dim)
    for i in range(device_indices.shape[0]):
        row = device_indices[i]
        assert (
            row.unique().numel() == row.numel()
        ), "Duplicate indices found in device output"

    # 2) Gather values using device_indices, compute cosine similarity with golden_values
    device_gathered = torch.gather(input_tensor, -1, device_indices)
    golden_gathered = torch.gather(input_tensor, -1, golden_indices)
    cos_sim = torch.nn.functional.cosine_similarity(
        device_gathered.flatten().unsqueeze(0).float(),
        golden_gathered.flatten().unsqueeze(0).float(),
    )
    assert cos_sim > 0.99, f"Cosine similarity: {cos_sim.item()} (required > 0.99)"


def topk_both_comparator(device_output, golden_output, args, kwargs):
    """Comparator for topk operation returning both values and indices."""
    device_values, device_indices = device_output
    golden_values, _ = golden_output
    input_tensor = args[0]

    # Move device outputs from XLA to CPU for comparison
    device_values = device_values.cpu()
    device_indices = device_indices.cpu()

    # 1) PCC between golden_values and device_values
    pcc = compute_pcc(golden_values, device_values)
    assert pcc > 0.99, f"PCC between golden and device values: {pcc} (required > 0.99)"

    # 2) Assert device_indices has no duplicate elements (per row, along last dim)
    for i in range(device_indices.shape[0]):
        row = device_indices[i]
        assert (
            row.unique().numel() == row.numel()
        ), "Duplicate indices found in device output"

    # 3) Gather values using device_indices, compute cosine similarity with golden_values
    gathered = torch.gather(input_tensor, -1, device_indices)
    cos_sim = torch.nn.functional.cosine_similarity(
        gathered.flatten().unsqueeze(0).float(),
        golden_values.flatten().unsqueeze(0).float(),
    )
    assert cos_sim > 0.99, f"Cosine similarity: {cos_sim.item()} (required > 0.99)"


@pytest.mark.nightly
@pytest.mark.single_device
@pytest.mark.record_test_properties(
    category=Category.OP_TEST,
    torch_op_name="torch.topk",
)
@pytest.mark.parametrize(
    ["input_shape", "k"],
    [
        ((1, 10), 5),
        ((1, 20), 5),
        ((1, 30), 5),
        ((1, 40), 5),
        ((1, 50000), 100),
        pytest.param(
            (1, 8400),
            300,
            marks=pytest.mark.xfail(
                reason="Bad PCC due to ttnn sort bug for greater than 256 elements - https://github.com/tenstorrent/tt-xla/issues/1797"
            ),
        ),
    ],
)
def test_topk_indices(input_shape: tuple, k: int):
    """Test topk operation returning indices."""

    class TopKIndices(torch.nn.Module):
        def __init__(self, k):
            super().__init__()
            self.k = k

        def forward(self, x):
            values, indices = torch.topk(x, self.k)
            return values, indices

    model = TopKIndices(k)

    run_op_test_with_random_inputs(
        model,
        [input_shape],
        dtype=torch.float32,
        framework=Framework.TORCH,
        custom_comparator=topk_indices_comparator,
    )


@pytest.mark.nightly
@pytest.mark.single_device
@pytest.mark.record_test_properties(
    category=Category.OP_TEST,
    torch_op_name="torch.topk",
)
@pytest.mark.parametrize(
    ["input_shape", "k"],
    [
        ((1, 10), 5),
        ((1, 20), 5),
        ((1, 30), 5),
        ((1, 40), 5),
    ],
)
def test_topk_values(input_shape: tuple, k: int):
    """Test topk operation returning only values."""

    class TopKValues(torch.nn.Module):
        def __init__(self, k):
            super().__init__()
            self.k = k

        def forward(self, x):
            return torch.topk(x, self.k)[0]

    model = TopKValues(k)

    run_op_test_with_random_inputs(
        model, [input_shape], dtype=torch.float32, framework=Framework.TORCH
    )


@pytest.mark.nightly
@pytest.mark.single_device
@pytest.mark.record_test_properties(
    category=Category.OP_TEST,
    torch_op_name="torch.topk",
)
@pytest.mark.parametrize(
    ["input_shape", "k"],
    [
        ((1, 10), 5),
        ((1, 20), 5),
        ((1, 30), 5),
        ((1, 40), 5),
    ],
)
def test_topk_both(input_shape: tuple, k: int):
    """Test topk operation returning both values and indices."""

    class TopKBoth(torch.nn.Module):
        def __init__(self, k):
            super().__init__()
            self.k = k

        def forward(self, x):
            values, indices = torch.topk(x, self.k)
            return values, indices

    model = TopKBoth(k)

    run_op_test_with_random_inputs(
        model, [input_shape], dtype=torch.float32, framework=Framework.TORCH
    )
