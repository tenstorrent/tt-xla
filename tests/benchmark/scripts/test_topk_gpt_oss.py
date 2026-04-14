# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Isolated topk test reproducing the GPT-OSS MoE router configuration.

Extracted from TTNN op trace:
  #678 typecast  | IN global_id=213 → OUT 272x32 dt=2 (bfloat16)
  #680 ttnn.topk | IN global_id=214 (272x32 bf16), k=4
       → values  global_id=215: 272x4 bf16 (→ softmax #688)
       → indices global_id=216: 272x4 int32 (→ typecast #682 + reshape + concat)

This is the MoE expert router: 32 experts, top-4 selection.
Per-device shape on Galaxy (32 devices, DP=4, TP=8): [272, 32] → topk(k=4).

Usage (from repo root, with venv activated):
    pytest -svv tests/benchmark/scripts/test_topk_gpt_oss.py
    python tests/benchmark/scripts/test_topk_gpt_oss.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_BENCH_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_BENCH_ROOT))

from benchmark.utils import compute_pcc  # noqa: E402
from infra import Framework, run_op_test_with_random_inputs  # noqa: E402


def _topk_both_comparator(device_output, golden_output, args, kwargs):
    """Compare topk values via PCC and indices via gathered cosine similarity."""
    device_values, device_indices = device_output
    golden_values, golden_indices = golden_output
    input_tensor = args[0]

    device_values = device_values.cpu()
    device_indices = device_indices.cpu()

    pcc = compute_pcc(golden_values, device_values)
    assert pcc > 0.99, f"PCC between golden and device values: {pcc} (required > 0.99)"

    for i in range(device_indices.shape[0]):
        row = device_indices[i]
        assert (
            row.unique().numel() == row.numel()
        ), f"Duplicate indices found in device output row {i}"

    gathered = torch.gather(input_tensor, -1, device_indices)
    cos_sim = torch.nn.functional.cosine_similarity(
        gathered.flatten().unsqueeze(0).float(),
        golden_values.flatten().unsqueeze(0).float(),
    )
    assert cos_sim > 0.99, f"Cosine similarity: {cos_sim.item()} (required > 0.99)"


def _topk_values_comparator(device_output, golden_output, args, kwargs):
    """Compare topk values only via PCC."""
    device_values = device_output.cpu() if hasattr(device_output, "cpu") else device_output
    golden_values = golden_output
    pcc = compute_pcc(golden_values, device_values)
    assert pcc > 0.99, f"PCC between golden and device values: {pcc} (required > 0.99)"


def _topk_indices_comparator(device_output, golden_output, args, kwargs):
    """Compare topk indices via gathered cosine similarity."""
    device_indices = device_output.cpu() if hasattr(device_output, "cpu") else device_output
    golden_indices = golden_output
    input_tensor = args[0]

    for i in range(device_indices.shape[0]):
        row = device_indices[i]
        assert (
            row.unique().numel() == row.numel()
        ), f"Duplicate indices found in device output row {i}"

    device_gathered = torch.gather(input_tensor, -1, device_indices)
    golden_gathered = torch.gather(input_tensor, -1, golden_indices)
    cos_sim = torch.nn.functional.cosine_similarity(
        device_gathered.flatten().unsqueeze(0).float(),
        golden_gathered.flatten().unsqueeze(0).float(),
    )
    assert cos_sim > 0.99, f"Cosine similarity: {cos_sim.item()} (required > 0.99)"


# --- GPT-OSS MoE router topk: [272, 32] → top-4 ---

GPT_OSS_INPUT_SHAPE = (272, 32)
GPT_OSS_K = 4


class TopKBoth(torch.nn.Module):
    def __init__(self, k: int):
        super().__init__()
        self.k = k

    def forward(self, x):
        values, indices = torch.topk(x, self.k)
        return values, indices


class TopKValues(torch.nn.Module):
    def __init__(self, k: int):
        super().__init__()
        self.k = k

    def forward(self, x):
        values, _ = torch.topk(x, self.k)
        return values


class TopKIndices(torch.nn.Module):
    def __init__(self, k: int):
        super().__init__()
        self.k = k

    def forward(self, x):
        _, indices = torch.topk(x, self.k)
        return indices


@pytest.mark.single_device
@pytest.mark.parametrize(
    ["input_shape", "k", "dtype", "minval", "maxval"],
    [
        pytest.param(GPT_OSS_INPUT_SHAPE, GPT_OSS_K, torch.bfloat16, 0.0, 1.0, id="gpt_oss_moe_bf16"),
        pytest.param(GPT_OSS_INPUT_SHAPE, GPT_OSS_K, torch.float32, 0.0, 1.0, id="gpt_oss_moe_fp32"),
        pytest.param(GPT_OSS_INPUT_SHAPE, GPT_OSS_K, torch.bfloat16, -10.0, -0.01, id="gpt_oss_moe_bf16_allneg"),
        pytest.param(GPT_OSS_INPUT_SHAPE, GPT_OSS_K, torch.float32, -10.0, -0.01, id="gpt_oss_moe_fp32_allneg"),
    ],
)
def test_topk_gpt_oss_both(input_shape: tuple, k: int, dtype: torch.dtype, minval: float, maxval: float):
    """TopK returning both values and indices — GPT-OSS MoE router shape."""
    run_op_test_with_random_inputs(
        TopKBoth(k),
        [input_shape],
        minval=minval,
        maxval=maxval,
        dtype=dtype,
        framework=Framework.TORCH,
        custom_comparator=_topk_both_comparator,
    )


@pytest.mark.single_device
@pytest.mark.parametrize(
    ["input_shape", "k", "dtype", "minval", "maxval"],
    [
        pytest.param(GPT_OSS_INPUT_SHAPE, GPT_OSS_K, torch.bfloat16, 0.0, 1.0, id="gpt_oss_moe_bf16"),
        pytest.param(GPT_OSS_INPUT_SHAPE, GPT_OSS_K, torch.float32, 0.0, 1.0, id="gpt_oss_moe_fp32"),
        pytest.param(GPT_OSS_INPUT_SHAPE, GPT_OSS_K, torch.bfloat16, -10.0, -0.01, id="gpt_oss_moe_bf16_allneg"),
        pytest.param(GPT_OSS_INPUT_SHAPE, GPT_OSS_K, torch.float32, -10.0, -0.01, id="gpt_oss_moe_fp32_allneg"),
    ],
)
def test_topk_gpt_oss_values(input_shape: tuple, k: int, dtype: torch.dtype, minval: float, maxval: float):
    """TopK returning values only — GPT-OSS MoE router shape."""
    run_op_test_with_random_inputs(
        TopKValues(k),
        [input_shape],
        minval=minval,
        maxval=maxval,
        dtype=dtype,
        framework=Framework.TORCH,
    )


@pytest.mark.single_device
@pytest.mark.parametrize(
    ["input_shape", "k", "dtype", "minval", "maxval"],
    [
        pytest.param(GPT_OSS_INPUT_SHAPE, GPT_OSS_K, torch.bfloat16, 0.0, 1.0, id="gpt_oss_moe_bf16"),
        pytest.param(GPT_OSS_INPUT_SHAPE, GPT_OSS_K, torch.float32, 0.0, 1.0, id="gpt_oss_moe_fp32"),
        pytest.param(GPT_OSS_INPUT_SHAPE, GPT_OSS_K, torch.bfloat16, -10.0, -0.01, id="gpt_oss_moe_bf16_allneg"),
        pytest.param(GPT_OSS_INPUT_SHAPE, GPT_OSS_K, torch.float32, -10.0, -0.01, id="gpt_oss_moe_fp32_allneg"),
    ],
)
def test_topk_gpt_oss_indices(input_shape: tuple, k: int, dtype: torch.dtype, minval: float, maxval: float):
    """TopK returning indices only — GPT-OSS MoE router shape."""
    run_op_test_with_random_inputs(
        TopKIndices(k),
        [input_shape],
        minval=minval,
        maxval=maxval,
        dtype=dtype,
        framework=Framework.TORCH,
        custom_comparator=_topk_indices_comparator,
    )


if __name__ == "__main__":
    pytest.main([__file__, "-svv"])
