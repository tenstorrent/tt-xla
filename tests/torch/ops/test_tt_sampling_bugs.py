# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Regression tests for TT device bugs affecting non-greedy vLLM sampling.

These tests go through the torch.compile / TT compilation flow (via
run_op_test with Framework.TORCH) rather than eager execution, per
recommendation from Het Shah — eager execution does not insert the composite
op lowerings used in production.

Bug 1 (topk wrong indices): Per Het Shah, ttnn.topk does not have the same
index bug as ttnn.sort. The correct test uses gathered values for comparison,
not exact index match (topk ordering is non-deterministic).

Bug 2 (gather int64): Original eager test showed mismatch. Converted to
compiled execution per Het Shah — if this still fails it indicates a bug in
the stablehlo.gather → ttnn.gather lowering in tt-mlir.

Related: vllm_sampling_tt_bugs.md, tt-xla issue #4329
"""

import pytest
import torch
from infra import Framework, run_op_test

SEED = 42


# ---------------------------------------------------------------------------
# topk: correct values selected, ordering non-deterministic
# ---------------------------------------------------------------------------


def topk_values_comparator(device_output, golden_output, args, kwargs):
    """Correct topk comparator: gathered values must match, not exact indices.

    torch.topk does not guarantee ordering among the top-k elements.
    CPU may return [0,1,2,3], device [3,2,1,0] — both are valid.
    """
    device_vals, device_idx = device_output
    golden_vals, golden_idx = golden_output
    input_tensor = args[0]

    device_idx = device_idx.cpu()
    golden_idx = golden_idx.cpu()

    device_gathered = torch.gather(input_tensor, -1, device_idx)
    golden_gathered = torch.gather(input_tensor, -1, golden_idx)
    cos_sim = torch.nn.functional.cosine_similarity(
        device_gathered.flatten().unsqueeze(0).float(),
        golden_gathered.flatten().unsqueeze(0).float(),
    )
    idx_match = (device_idx == golden_idx).float().mean().item()
    k = device_idx.shape[-1]
    print(
        f"\n  values_cos_sim={cos_sim.item():.6f}"
        f"  index_exact_match={idx_match:.3f} ({int(idx_match * k)}/{k})"
        f"  (ordering non-deterministic — only values matter)"
    )
    assert cos_sim > 0.99, f"topk gathered values wrong: cos_sim={cos_sim.item():.6f}"


@pytest.mark.single_device
@pytest.mark.parametrize(
    "shape,k",
    [
        pytest.param((1, 32768), 32, id="32768-k32-vllm-chunk"),
        pytest.param((1, 32768), 64, id="32768-k64"),
        pytest.param((1, 16384), 32, id="16384-k32"),
        pytest.param((1, 65536), 32, id="65536-k32"),
    ],
)
def test_topk_values_correct(shape, k):
    """torch.topk selects the correct top-k values (ordering may differ)."""

    class TopKBoth(torch.nn.Module):
        def __init__(self, k):
            super().__init__()
            self.k = k

        def forward(self, x):
            return torch.topk(x, self.k, dim=-1)

    torch.manual_seed(SEED)
    x_cpu = torch.randn(*shape, dtype=torch.float32)

    run_op_test(
        TopKBoth(k),
        [x_cpu],
        framework=Framework.TORCH,
        custom_comparator=topk_values_comparator,
    )


# ---------------------------------------------------------------------------
# gather int64: compiled execution (per Het Shah recommendation)
# ---------------------------------------------------------------------------


def gather_int64_comparator(device_output, golden_output, args, kwargs):
    """Comparator for torch.gather on int64 index tensors.

    The gathered value must be an exact int64 match — any mismatch indicates
    a bug in the stablehlo.gather → ttnn.gather lowering in tt-mlir.
    """
    device_value = device_output.cpu().item()
    golden_value = golden_output.item()
    local_idx = args[1].item()
    print(
        f"\n  local_idx={local_idx}  "
        f"cpu_token={golden_value}  dev_token={device_value}  "
        f"match={golden_value == device_value}"
    )
    assert golden_value == device_value, (
        f"gather int64 mismatch: cpu={golden_value} dev={device_value} "
        f"(local_idx={local_idx}) — stablehlo.gather lowering bug in tt-mlir"
    )


@pytest.mark.single_device
@pytest.mark.parametrize(
    "candidates,vocab_size",
    [
        pytest.param(64, 50272, id="opt125m"),
        pytest.param(128, 128256, id="llama"),
    ],
)
def test_gather_int64_correctness(candidates, vocab_size):
    """torch.gather on int64 tensors via compiled execution.

    Uses run_op_test (Framework.TORCH) so the op goes through torch.compile
    and the TT stablehlo.gather lowering path, per Het Shah's recommendation.
    If this fails it indicates a bug in the lowering, not in eager execution.
    """

    class GatherInt64(torch.nn.Module):
        def forward(self, idx, local):
            return torch.gather(idx, 1, local)

    torch.manual_seed(SEED)
    idx_cpu = torch.randperm(vocab_size, dtype=torch.int64)[:candidates].unsqueeze(0)
    local_cpu = torch.randint(0, candidates, (1, 1), dtype=torch.int64)

    run_op_test(
        GatherInt64(),
        [idx_cpu, local_cpu],
        framework=Framework.TORCH,
        custom_comparator=gather_int64_comparator,
    )
