# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the shared weight-cache infrastructure.

These tests are pure-CPU and don't touch HF or any actual model — they verify
the small helpers used by every cache builder. Run inside the docker
container with:

    pytest -v tests/infra/weight_cache/test_weight_cache.py
"""
import pytest
import torch
from infra.weight_cache import (
    cache_dir_for,
    fp8_blockwise_dequant,
    group_keys_by_shard,
    has_cache,
    is_fp8,
    maybe_dequant,
    safe_open_hf,
)

# ---------------------------------------------------------------------------
# paths.py
# ---------------------------------------------------------------------------


def test_cache_dir_for_default_variant(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    p = cache_dir_for("deepseek-ai/DeepSeek-V3.1", 4)
    assert p == (
        tmp_path / "tt_xla_weight_cache" / "deepseek-ai--DeepSeek-V3.1_4layers_bf16"
    )


def test_cache_dir_for_stacked_variant(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    p = cache_dir_for("zai-org/GLM-4.7", 92, variant="stacked")
    assert p == (tmp_path / "tt_xla_weight_cache" / "zai-org--GLM-4.7_92layers_stacked")


def test_safe_open_hf_inside_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    inside = tmp_path / "sub" / "ok.json"
    inside.parent.mkdir(parents=True)
    inside.write_text("{}")
    with safe_open_hf(inside) as fh:
        assert fh.read() == "{}"


def test_safe_open_hf_rejects_outside_cache(tmp_path, monkeypatch):
    cache = tmp_path / "cache"
    cache.mkdir()
    monkeypatch.setenv("HF_HOME", str(cache))
    outside = tmp_path / "outside.json"
    outside.write_text("{}")
    with pytest.raises(ValueError, match="outside HF cache"):
        safe_open_hf(outside)


def test_has_cache_nonexistent(tmp_path):
    assert not has_cache(tmp_path / "does_not_exist")


def test_has_cache_empty_dir(tmp_path):
    assert not has_cache(tmp_path)


def test_has_cache_only_other_files(tmp_path):
    (tmp_path / "readme.txt").write_text("not a cache")
    assert not has_cache(tmp_path)


def test_has_cache_with_safetensors(tmp_path):
    (tmp_path / "layer_0000.safetensors").write_bytes(b"")
    assert has_cache(tmp_path)


# ---------------------------------------------------------------------------
# shards.py
# ---------------------------------------------------------------------------


def test_group_keys_by_shard_basic():
    weight_map = {
        "a.weight": "shard-001.safetensors",
        "b.weight": "shard-001.safetensors",
        "c.weight": "shard-002.safetensors",
    }
    out = group_keys_by_shard(["a.weight", "b.weight", "c.weight"], weight_map)
    assert out == {
        "shard-001.safetensors": ["a.weight", "b.weight"],
        "shard-002.safetensors": ["c.weight"],
    }


def test_group_keys_by_shard_subset():
    weight_map = {
        "a.weight": "shard-001.safetensors",
        "b.weight": "shard-001.safetensors",
        "c.weight": "shard-002.safetensors",
    }
    out = group_keys_by_shard(["a.weight", "c.weight"], weight_map)
    assert out == {
        "shard-001.safetensors": ["a.weight"],
        "shard-002.safetensors": ["c.weight"],
    }


# ---------------------------------------------------------------------------
# dequant.py
# ---------------------------------------------------------------------------


def test_is_fp8():
    assert is_fp8(torch.zeros(1, dtype=torch.float8_e4m3fn))
    assert is_fp8(torch.zeros(1, dtype=torch.float8_e5m2))
    assert not is_fp8(torch.zeros(1, dtype=torch.bfloat16))
    assert not is_fp8(torch.zeros(1, dtype=torch.float32))


def test_maybe_dequant_passthrough_for_bf16():
    t = torch.randn(4, 4, dtype=torch.bfloat16)
    assert maybe_dequant(t, None) is t


def test_maybe_dequant_passthrough_when_no_scale():
    t = torch.zeros(2, 2, dtype=torch.float8_e4m3fn)
    assert maybe_dequant(t, None) is t


def test_fp8_blockwise_dequant_single_block():
    # 2x2 weight = one 2x2 block, single scalar scale
    weight = torch.tensor([[1.0, 2.0], [4.0, 8.0]], dtype=torch.float8_e4m3fn)
    scale_inv = torch.tensor([[2.0]], dtype=torch.float32)
    out = fp8_blockwise_dequant(weight, scale_inv, block_size=2)
    assert out.shape == (2, 2)
    assert out.dtype == torch.bfloat16
    expected = torch.tensor([[2.0, 4.0], [8.0, 16.0]], dtype=torch.bfloat16)
    assert torch.equal(out, expected)


def test_fp8_blockwise_dequant_two_blocks_per_row():
    # 2x4 weight = 1x2 grid of 2x2 blocks
    weight = torch.tensor(
        [[1.0, 2.0, 4.0, 8.0], [4.0, 8.0, 1.0, 2.0]],
        dtype=torch.float8_e4m3fn,
    )
    scale_inv = torch.tensor([[2.0, 0.5]], dtype=torch.float32)
    out = fp8_blockwise_dequant(weight, scale_inv, block_size=2)
    expected = torch.tensor(
        [[2.0, 4.0, 2.0, 4.0], [8.0, 16.0, 0.5, 1.0]],
        dtype=torch.bfloat16,
    )
    assert torch.equal(out, expected)


def test_fp8_blockwise_dequant_with_padding():
    # 3x3 weight padded to 4x4 (block_size=2). The padded zeros must be
    # multiplied by the right scale but then sliced away — the result is 3x3.
    weight = torch.tensor(
        [[1.0, 2.0, 4.0], [4.0, 8.0, 1.0], [2.0, 4.0, 1.0]],
        dtype=torch.float8_e4m3fn,
    )
    scale_inv = torch.tensor([[1.0, 2.0], [4.0, 8.0]], dtype=torch.float32)
    out = fp8_blockwise_dequant(weight, scale_inv, block_size=2)
    assert out.shape == (3, 3)
    expected = torch.tensor(
        [[1.0, 2.0, 8.0], [4.0, 8.0, 2.0], [8.0, 16.0, 8.0]],
        dtype=torch.bfloat16,
    )
    assert torch.equal(out, expected)


def test_maybe_dequant_applies_for_fp8_with_scale():
    weight = torch.tensor([[1.0, 2.0], [4.0, 8.0]], dtype=torch.float8_e4m3fn)
    scale_inv = torch.tensor([[2.0]], dtype=torch.float32)
    out = maybe_dequant(weight, scale_inv)
    assert out.dtype == torch.bfloat16
    expected = torch.tensor([[2.0, 4.0], [8.0, 16.0]], dtype=torch.bfloat16)
    assert torch.equal(out, expected)
