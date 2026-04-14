# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Test ttnn.typecast ui16→si32 using real topk indices from npy dumps.

Reproduces the GPT-OSS MoE pipeline op:
  %164 = "ttnn.typecast"(%indices) <{dtype = si32}>
       : (tensor<272x4xui16>) -> tensor<272x4xsi32>

The topk indices (small values 0..31) should survive a uint16→int32 cast
unchanged. The trace showed all-zero output from this typecast — this test
checks whether the bug reproduces on a single device.

Usage (from repo root, with venv activated):
    pytest -svv tests/benchmark/scripts/test_typecast_ui16_si32.py
    python tests/benchmark/scripts/test_typecast_ui16_si32.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_BENCH_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_BENCH_ROOT))

from infra import Framework, run_op_test  # noqa: E402

DEFAULT_DUMP_DIR = (
    _BENCH_ROOT
    / "modules"
    / "gpt_oss_input_sharding_dbg"
    / "topk_dump"
)


def _discover_dump_dir() -> Path:
    """Return the topk dump directory, checking env override first."""
    import os

    env_dir = os.environ.get("TOPK_DUMP_DIR")
    if env_dir:
        p = Path(env_dir)
        if p.is_dir():
            return p
    if DEFAULT_DUMP_DIR.is_dir():
        return DEFAULT_DUMP_DIR
    pytest.skip(f"No topk dump directory found at {DEFAULT_DUMP_DIR}")


def _load_indices_for_device(dump_dir: Path, seq: int, dev: int) -> np.ndarray:
    """Load topk indices npy for a given sequence number and device."""
    path = dump_dir / f"topk_{seq}_indices_dev{dev}.npy"
    if not path.exists():
        pytest.skip(f"Missing dump file: {path}")
    return np.load(str(path))


def _discover_sequences(dump_dir: Path) -> list[int]:
    """Find all topk sequence numbers that have meta files."""
    seqs = []
    for p in sorted(dump_dir.glob("topk_*_meta.json")):
        stem = p.stem
        seq = int(stem.split("_")[1])
        seqs.append(seq)
    return sorted(set(seqs))


def _discover_devices(dump_dir: Path, seq: int) -> list[int]:
    """Find all device IDs for a given topk sequence."""
    devs = []
    for p in sorted(dump_dir.glob(f"topk_{seq}_indices_dev*.npy")):
        stem = p.stem
        dev = int(stem.rsplit("dev", 1)[1])
        devs.append(dev)
    return sorted(devs)


class TypecastToInt32(torch.nn.Module):
    """Simple module that casts input to int32 — mirrors ttnn.typecast(ui16→si32)."""

    def forward(self, x):
        return x.to(torch.int32)


def _typecast_comparator(device_output, golden_output, args, kwargs):
    """Verify typecast output matches golden exactly."""
    dev = device_output.cpu() if hasattr(device_output, "cpu") else device_output
    gold = golden_output.cpu() if hasattr(golden_output, "cpu") else golden_output

    nonzero_input = (args[0] != 0).sum().item()
    nonzero_dev = (dev != 0).sum().item()
    nonzero_gold = (gold != 0).sum().item()

    print(f"  input  nonzero elements: {nonzero_input} / {args[0].numel()}")
    print(f"  golden nonzero elements: {nonzero_gold} / {gold.numel()}")
    print(f"  device nonzero elements: {nonzero_dev} / {dev.numel()}")

    if nonzero_dev == 0 and nonzero_gold > 0:
        print("  BUG REPRODUCED: device output is all zeros but golden is not!")

    exact_match = torch.equal(dev.int(), gold.int())
    print(f"  exact match: {exact_match}")
    assert exact_match, (
        f"Typecast output mismatch: "
        f"device has {nonzero_dev} nonzero vs golden {nonzero_gold} nonzero"
    )


def _get_test_cases() -> list[tuple[int, int]]:
    """Build (seq, dev) pairs for parametrize. Returns at least one fallback."""
    dump_dir = DEFAULT_DUMP_DIR
    if not dump_dir.is_dir():
        return [(1, 0)]
    seqs = _discover_sequences(dump_dir)
    if not seqs:
        return [(1, 0)]
    cases = []
    for seq in seqs:
        devs = _discover_devices(dump_dir, seq)
        for dev in devs[:1]:
            cases.append((seq, dev))
    return cases


@pytest.mark.single_device
@pytest.mark.parametrize(
    ["seq", "dev"],
    [pytest.param(s, d, id=f"topk{s}_dev{d}") for s, d in _get_test_cases()],
)
def test_typecast_ui16_to_si32_from_dumps(seq: int, dev: int):
    """Typecast ui16→si32 using real topk indices from npy dumps."""
    dump_dir = _discover_dump_dir()
    indices_i32 = _load_indices_for_device(dump_dir, seq, dev)

    print(f"\n=== topk #{seq} dev {dev} ===")
    print(f"  loaded indices shape={indices_i32.shape} dtype={indices_i32.dtype}")
    print(f"  value range: [{indices_i32.min()}, {indices_i32.max()}]")
    print(f"  nonzero: {np.count_nonzero(indices_i32)} / {indices_i32.size}")
    print(f"  first row: {indices_i32[0]}")

    input_tensor = torch.from_numpy(indices_i32.astype(np.int16)).to(torch.int16)
    print(f"  input tensor (int16): shape={input_tensor.shape} dtype={input_tensor.dtype}")
    print(f"  input first row: {input_tensor[0].tolist()}")

    run_op_test(
        TypecastToInt32(),
        [input_tensor],
        framework=Framework.TORCH,
        custom_comparator=_typecast_comparator,
    )


@pytest.mark.single_device
def test_typecast_ui16_to_si32_synthetic():
    """Typecast ui16→si32 with synthetic data matching GPT-OSS topk shape."""
    shape = (272, 4)
    indices = torch.randint(0, 32, shape, dtype=torch.int16)

    print(f"\n=== synthetic test ===")
    print(f"  shape={shape} dtype={indices.dtype}")
    print(f"  value range: [{indices.min().item()}, {indices.max().item()}]")
    print(f"  first row: {indices[0].tolist()}")

    run_op_test(
        TypecastToInt32(),
        [indices],
        framework=Framework.TORCH,
        custom_comparator=_typecast_comparator,
    )


@pytest.mark.single_device
@pytest.mark.parametrize(
    ["shape", "lo", "hi", "desc"],
    [
        pytest.param((272, 4), 0, 32, "topk_range", id="topk_range"),
        pytest.param((272, 4), 0, 1, "near_zero", id="near_zero"),
        pytest.param((272, 4), 100, 200, "mid_range", id="mid_range"),
        pytest.param((272, 4), 0, 32767, "full_i16_range", id="full_i16_range"),
        pytest.param((16, 4), 0, 32, "small_shape", id="small_shape"),
        pytest.param((1088, 4), 0, 32, "gathered_shape", id="gathered_shape"),
    ],
)
def test_typecast_ui16_to_si32_sweep(shape: tuple, lo: int, hi: int, desc: str):
    """Sweep typecast ui16→si32 across various shapes and value ranges."""
    indices = torch.randint(lo, hi, shape, dtype=torch.int16)

    print(f"\n=== sweep: {desc} ===")
    print(f"  shape={shape} range=[{lo},{hi}) dtype={indices.dtype}")
    print(f"  first row: {indices[0].tolist()}")

    run_op_test(
        TypecastToInt32(),
        [indices],
        framework=Framework.TORCH,
        custom_comparator=_typecast_comparator,
    )


if __name__ == "__main__":
    pytest.main([__file__, "-svv"])
