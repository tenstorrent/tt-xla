# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""CPU unit tests for the MRv2 runner get_kv_cache_spec.

``TTModelRunnerV2.get_kv_cache_spec`` (see vllm_tt/model_runner_v2.py) walks the
attention layers registered in the static forward context and emits a KVCacheSpec
per layer. The layers normally come from a loaded model; here they are faked as
bare ``Attention`` instances placed in a duck-typed vllm_config, so the
spec-selection logic runs on cpu with no model or TT hardware.

They pin the selection TT owns: full vs sliding-window specs, cross-layer KV
sharing (skipped + recorded), and encoder-only layers (no KV cache).
"""

from types import SimpleNamespace

import pytest
import torch
from vllm.model_executor.layers.attention.attention import Attention
from vllm.v1.attention.backend import AttentionType
from vllm.v1.kv_cache_interface import FullAttentionSpec, SlidingWindowSpec
from vllm_tt.model_runner_v2 import TTModelRunnerV2

BLOCK_SIZE = 32


def fake_attn(
    num_kv_heads=8,
    head_size=64,
    attn_type=AttentionType.DECODER,
    sliding_window=None,
    kv_sharing_target_layer_name=None,
):
    # Bypass the heavy Attention.__init__; only these attrs are read.
    m = object.__new__(Attention)
    m.num_kv_heads = num_kv_heads
    m.head_size = head_size
    m.attn_type = attn_type
    m.sliding_window = sliding_window
    m.kv_sharing_target_layer_name = kv_sharing_target_layer_name
    return m


def make_runner(layers, cache_dtype="auto"):
    r = object.__new__(TTModelRunnerV2)
    r.kv_cache_spec_dtype = torch.bfloat16
    r.shared_kv_cache_layers = {}
    r.vllm_config = SimpleNamespace(
        compilation_config=SimpleNamespace(static_forward_context=layers),
        cache_config=SimpleNamespace(block_size=BLOCK_SIZE, cache_dtype=cache_dtype),
    )
    return r


@pytest.mark.push
@pytest.mark.cpu
def test_full_attention_layers():
    r = make_runner({"l0": fake_attn(), "l1": fake_attn(num_kv_heads=4, head_size=128)})
    spec = r.get_kv_cache_spec()

    assert set(spec) == {"l0", "l1"}
    assert isinstance(spec["l0"], FullAttentionSpec)
    assert spec["l0"].num_kv_heads == 8
    assert spec["l0"].head_size == 64
    assert spec["l0"].block_size == BLOCK_SIZE
    assert spec["l0"].dtype == torch.bfloat16
    assert spec["l1"].num_kv_heads == 4
    assert spec["l1"].head_size == 128


@pytest.mark.push
@pytest.mark.cpu
def test_sliding_window_layer():
    r = make_runner({"l0": fake_attn(sliding_window=256)})
    spec = r.get_kv_cache_spec()
    assert isinstance(spec["l0"], SlidingWindowSpec)
    assert spec["l0"].sliding_window == 256


@pytest.mark.push
@pytest.mark.cpu
def test_kv_sharing_layer_skipped_and_recorded():
    r = make_runner(
        {
            "l0": fake_attn(),
            "l1": fake_attn(kv_sharing_target_layer_name="l0"),
        }
    )
    spec = r.get_kv_cache_spec()
    # The sharing layer gets no spec of its own; it is recorded as sharing l0.
    assert set(spec) == {"l0"}
    assert r.shared_kv_cache_layers == {"l1": "l0"}


@pytest.mark.push
@pytest.mark.cpu
def test_encoder_only_layer_has_no_kv_cache():
    r = make_runner(
        {
            "dec": fake_attn(),
            "enc": fake_attn(attn_type=AttentionType.ENCODER_ONLY),
        }
    )
    spec = r.get_kv_cache_spec()
    assert set(spec) == {"dec"}
