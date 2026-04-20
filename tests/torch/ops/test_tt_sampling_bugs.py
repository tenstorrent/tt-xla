# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Regression tests for TT device bugs affecting non-greedy vLLM sampling.

Bug 1: torch.topk returns correct VALUES but wrong INDICES on TT device
        for shape [batch, 32768], k=32 (the chunk size used in vLLM sampling).
        Discovered via test_sampling_pipeline.py: stage1 index exact-match=0.61.

Bug 2: torch.gather on int64 index tensors returns wrong values on TT device.
        Discovered via test_sampling_pipeline.py: stage5 cpu=74658 dev=74752.

Both bugs are in tt-metal / tt-mlir and must be fixed there.
These tests serve as reproducers to track when they are resolved.

Related: vllm_sampling_tt_bugs.md
"""

import pytest
import torch
import torch_xla.core.xla_model as xm
from infra import Framework, run_op_test

SEED = 42


@pytest.fixture
def device():
    return xm.xla_device()


# ---------------------------------------------------------------------------
# Bug 1: torch.topk returns wrong indices for [batch, 32768], k=32
# ---------------------------------------------------------------------------


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
def test_topk_index_correctness(shape, k, device):
    """torch.topk should return the same indices on TT device as on CPU.

    Known failure: (1, 32768), k=32 returns ~60% correct indices on TT.
    Values are correct (PCC > 0.99); only the index mapping is broken.
    """
    torch.manual_seed(SEED)
    x_cpu = torch.randn(*shape, dtype=torch.float32)
    x_dev = x_cpu.to(device)

    vals_cpu, idx_cpu = torch.topk(x_cpu, k=k, dim=-1)
    vals_dev, idx_dev = torch.topk(x_dev, k=k, dim=-1)

    vals_dev_cpu = vals_dev.cpu()
    idx_dev_cpu = idx_dev.cpu()

    # Values should match closely.
    vals_gathered_cpu = torch.gather(x_cpu, -1, idx_cpu)
    vals_gathered_dev = torch.gather(x_cpu, -1, idx_dev_cpu)
    cos_sim = torch.nn.functional.cosine_similarity(
        vals_gathered_cpu.flatten().unsqueeze(0).float(),
        vals_gathered_dev.flatten().unsqueeze(0).float(),
    )
    print(f"\n  values cosine_sim={cos_sim.item():.6f}")
    assert cos_sim > 0.99, f"topk values wrong: cosine_sim={cos_sim.item():.6f}"

    # Indices must be exact.
    idx_match = (idx_cpu == idx_dev_cpu).float().mean().item()
    print(f"  index exact-match={idx_match:.4f} ({int(idx_match * k)}/{k} correct)")
    assert idx_match > 0.99, (
        f"topk index mismatch: {idx_match:.4f} "
        f"({int(idx_match * k)}/{k} correct) — TT runtime bug"
    )


# ---------------------------------------------------------------------------
# Bug 2: torch.gather on int64 returns wrong values on TT device
# ---------------------------------------------------------------------------


def gather_int64_comparator(device_output, golden_output, args, kwargs):
    """Comparator for torch.gather on int64 index tensors.

    The gathered value must be an exact int64 match — any mismatch indicates
    the TT runtime bug on int64 gather.
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
        f"(local_idx={local_idx}) — TT runtime bug"
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
    """torch.gather on int64 index tensors should return correct values on TT.

    Known failure: gathering from [1, candidates] int64 with a [1, 1] int64
    index returns a wrong value on TT device.
    """

    class GatherInt64(torch.nn.Module):
        def forward(self, idx, local):
            return torch.gather(idx, 1, local)

    torch.manual_seed(SEED)
    # Simulate candidate_indices: global vocab positions of top candidates.
    idx_cpu = torch.randperm(vocab_size, dtype=torch.int64)[:candidates].unsqueeze(0)
    # Local sample index (as would come from argmax).
    local_cpu = torch.randint(0, candidates, (1, 1), dtype=torch.int64)

    run_op_test(
        GatherInt64(),
        [idx_cpu, local_cpu],
        framework=Framework.TORCH,
        custom_comparator=gather_int64_comparator,
    )
