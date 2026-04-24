# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Pytest tests for DeepSeek V4 Flash attention blocks using run_graph_test infrastructure.

Tests CSA (Compressed Sparse Attention) and HSA (Heavily Compressed Attention) blocks
with random weights on TT hardware.
"""
import sys
from pathlib import Path

import pytest
import torch
import torch_xla
import torch_xla.runtime as xr

# Add ds4 directory to path for imports
ds4_path = Path(__file__).parent
if str(ds4_path) not in sys.path:
    sys.path.insert(0, str(ds4_path))

from infra import Framework, run_graph_test
from infra.evaluators import ComparisonConfig, PccConfig
from modified_model import (
    ModelArgs,
    Attention,
    CSAAttention,
    HSAAttention,
    Compressor,
    Indexer,
)


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def small_model_args():
    """Small model args for fast testing."""
    return ModelArgs(
        dim=512,
        n_heads=8,
        head_dim=64,
        q_lora_rank=128,
        o_lora_rank=128,
        o_groups=2,
        index_n_heads=8,
        index_head_dim=64,
        index_topk=32,
        max_seq_len=256,
        max_batch_size=4,
        window_size=32,
        compress_ratios=(0, 0, 4, 128, 4, 128, 4, 0),
        rope_head_dim=16,
    )


@pytest.fixture
def medium_model_args():
    """Medium model args for more realistic testing."""
    return ModelArgs(
        dim=1024,
        n_heads=16,
        head_dim=128,
        q_lora_rank=256,
        o_lora_rank=256,
        o_groups=4,
        index_n_heads=16,
        index_head_dim=64,
        index_topk=64,
        max_seq_len=512,
        max_batch_size=4,
        window_size=64,
        compress_ratios=(0, 0, 4, 128, 4, 128, 4, 0),
        rope_head_dim=32,
    )


# ============================================================================
# CSA (Compressed Sparse Attention) Tests - compress_ratio=4
# ============================================================================


@pytest.mark.push
@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("seq_len", [32, 64])
def test_csa_attention_prefill(batch_size, seq_len, small_model_args):
    """
    Test CSA (Compressed Sparse Attention) in prefill mode.

    CSA uses compress_ratio=4 with an Indexer module for top-k selection
    of compressed KV positions.
    """
    xr.set_device_type("TT")

    args = small_model_args
    args.max_batch_size = batch_size
    args.max_seq_len = seq_len * 2

    # Create CSA attention (compress_ratio=4 with Indexer)
    attention = CSAAttention(args, layer_id=2)
    attention = attention.to(torch.bfloat16)
    attention.eval()

    # Create random inputs
    hidden_states = torch.randn(
        (batch_size, seq_len, args.dim),
        dtype=torch.bfloat16
    )

    comparison_config = ComparisonConfig(
        pcc=PccConfig(enabled=True, required_pcc=0.95),
    )

    run_graph_test(
        attention,
        [hidden_states, 0],  # x, start_pos
        framework=Framework.TORCH,
        comparison_config=comparison_config,
    )


@pytest.mark.push
@pytest.mark.parametrize("batch_size", [1, 2])
def test_csa_attention_decode(batch_size, small_model_args):
    """
    Test CSA attention in decode mode (single token generation).
    """
    xr.set_device_type("TT")

    args = small_model_args
    args.max_batch_size = batch_size
    prefill_len = 32

    attention = CSAAttention(args, layer_id=2)
    attention = attention.to(torch.bfloat16)
    attention.eval()

    # First do prefill to populate KV cache
    prefill_input = torch.randn(
        (batch_size, prefill_len, args.dim),
        dtype=torch.bfloat16
    )
    with torch.no_grad():
        _ = attention(prefill_input, start_pos=0)

    # Now test decode (single token)
    decode_input = torch.randn(
        (batch_size, 1, args.dim),
        dtype=torch.bfloat16
    )

    comparison_config = ComparisonConfig(
        pcc=PccConfig(enabled=True, required_pcc=0.95),
    )

    run_graph_test(
        attention,
        [decode_input, prefill_len],  # x, start_pos
        framework=Framework.TORCH,
        comparison_config=comparison_config,
    )


# ============================================================================
# HSA (Heavily Compressed Attention) Tests - compress_ratio=128
# ============================================================================


@pytest.mark.push
@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("seq_len", [128, 256])
def test_hsa_attention_prefill(batch_size, seq_len, small_model_args):
    """
    Test HSA (Heavily Compressed Attention) in prefill mode.

    HSA uses compress_ratio=128 without Indexer - it compresses all KV
    positions uniformly without top-k selection.
    """
    xr.set_device_type("TT")

    args = small_model_args
    args.max_batch_size = batch_size
    args.max_seq_len = seq_len * 2

    # Create HSA attention (compress_ratio=128 without Indexer)
    attention = HSAAttention(args, layer_id=3)
    attention = attention.to(torch.bfloat16)
    attention.eval()

    hidden_states = torch.randn(
        (batch_size, seq_len, args.dim),
        dtype=torch.bfloat16
    )

    comparison_config = ComparisonConfig(
        pcc=PccConfig(enabled=True, required_pcc=0.95),
    )

    run_graph_test(
        attention,
        [hidden_states, 0],
        framework=Framework.TORCH,
        comparison_config=comparison_config,
    )


@pytest.mark.push
@pytest.mark.parametrize("batch_size", [1, 2])
def test_hsa_attention_decode(batch_size, small_model_args):
    """
    Test HSA attention in decode mode.
    """
    xr.set_device_type("TT")

    args = small_model_args
    args.max_batch_size = batch_size
    prefill_len = 128  # HSA needs longer sequences due to compress_ratio=128

    attention = HSAAttention(args, layer_id=3)
    attention = attention.to(torch.bfloat16)
    attention.eval()

    # Prefill
    prefill_input = torch.randn(
        (batch_size, prefill_len, args.dim),
        dtype=torch.bfloat16
    )
    with torch.no_grad():
        _ = attention(prefill_input, start_pos=0)

    # Decode
    decode_input = torch.randn(
        (batch_size, 1, args.dim),
        dtype=torch.bfloat16
    )

    comparison_config = ComparisonConfig(
        pcc=PccConfig(enabled=True, required_pcc=0.95),
    )

    run_graph_test(
        attention,
        [decode_input, prefill_len],
        framework=Framework.TORCH,
        comparison_config=comparison_config,
    )


# ============================================================================
# Compressor Tests
# ============================================================================


@pytest.mark.push
@pytest.mark.parametrize("compress_ratio", [4, 128])
@pytest.mark.parametrize("batch_size", [1, 2])
def test_compressor(compress_ratio, batch_size, small_model_args):
    """
    Test the Compressor module that compresses KV cache.

    compress_ratio=4 uses overlapping windows (CSA style)
    compress_ratio=128 uses non-overlapping compression (HSA style)
    """
    xr.set_device_type("TT")

    args = small_model_args
    args.max_batch_size = batch_size
    seq_len = compress_ratio * 4  # Ensure enough tokens for compression

    compressor = Compressor(args, compress_ratio=compress_ratio, head_dim=args.head_dim)
    compressor = compressor.to(torch.bfloat16)

    # Setup required buffers
    compressor.kv_cache = torch.zeros(
        batch_size, args.max_seq_len // compress_ratio, args.head_dim,
        dtype=torch.bfloat16
    )
    from modified_model import precompute_freqs_cis
    compressor.freqs_cis = precompute_freqs_cis(
        args.rope_head_dim, args.max_seq_len, 0,
        args.rope_theta, args.rope_factor, args.beta_fast, args.beta_slow
    )

    hidden_states = torch.randn(
        (batch_size, seq_len, args.dim),
        dtype=torch.bfloat16
    )

    comparison_config = ComparisonConfig(
        pcc=PccConfig(enabled=True, required_pcc=0.95),
    )

    run_graph_test(
        compressor,
        [hidden_states, 0],
        framework=Framework.TORCH,
        comparison_config=comparison_config,
    )


# ============================================================================
# Indexer Tests (CSA component)
# ============================================================================


@pytest.mark.push
@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("seq_len", [32, 64])
def test_indexer(batch_size, seq_len, small_model_args):
    """
    Test the Indexer module used in CSA for top-k selection.

    The Indexer computes attention scores over compressed KV positions
    and returns top-k indices for sparse attention.
    """
    xr.set_device_type("TT")

    args = small_model_args
    args.max_batch_size = batch_size
    args.max_seq_len = seq_len * 2

    indexer = Indexer(args, compress_ratio=4)
    indexer = indexer.to(torch.bfloat16)

    # Enable raw scores for testing (avoids topk which may have numerical issues)
    indexer.return_raw_scores = True

    # Setup freqs_cis
    from modified_model import precompute_freqs_cis
    indexer.freqs_cis = precompute_freqs_cis(
        args.rope_head_dim, args.max_seq_len, 0,
        args.rope_theta, args.rope_factor, args.beta_fast, args.beta_slow
    )

    hidden_states = torch.randn(
        (batch_size, seq_len, args.dim),
        dtype=torch.bfloat16
    )
    qr = torch.randn(
        (batch_size, seq_len, args.q_lora_rank),
        dtype=torch.bfloat16
    )

    comparison_config = ComparisonConfig(
        pcc=PccConfig(enabled=True, required_pcc=0.95),
    )

    run_graph_test(
        indexer,
        [hidden_states, qr, 0, seq_len],  # x, qr, start_pos, offset
        framework=Framework.TORCH,
        comparison_config=comparison_config,
    )


# ============================================================================
# Sliding Window Attention Tests (baseline, no compression)
# ============================================================================


@pytest.mark.push
@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("seq_len", [32, 64])
def test_sliding_window_attention(batch_size, seq_len, small_model_args):
    """
    Test sliding window attention (compress_ratio=0).

    This is the baseline attention without any KV compression.
    """
    xr.set_device_type("TT")

    args = small_model_args
    args.max_batch_size = batch_size
    args.max_seq_len = seq_len * 2

    # layer_id=0 has compress_ratio=0 (sliding window only)
    attention = Attention(layer_id=0, args=args)
    attention = attention.to(torch.bfloat16)
    attention.eval()

    hidden_states = torch.randn(
        (batch_size, seq_len, args.dim),
        dtype=torch.bfloat16
    )

    comparison_config = ComparisonConfig(
        pcc=PccConfig(enabled=True, required_pcc=0.98),
    )

    run_graph_test(
        attention,
        [hidden_states, 0],
        framework=Framework.TORCH,
        comparison_config=comparison_config,
    )


# ============================================================================
# Nightly Tests (larger configs)
# ============================================================================


@pytest.mark.nightly
@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("seq_len", [128, 256, 512])
def test_csa_attention_nightly(batch_size, seq_len, medium_model_args):
    """Nightly test for CSA with larger configurations."""
    xr.set_device_type("TT")

    args = medium_model_args
    args.max_batch_size = batch_size
    args.max_seq_len = seq_len * 2

    attention = CSAAttention(args, layer_id=2)
    attention = attention.to(torch.bfloat16)
    attention.eval()

    hidden_states = torch.randn(
        (batch_size, seq_len, args.dim),
        dtype=torch.bfloat16
    )

    comparison_config = ComparisonConfig(
        pcc=PccConfig(enabled=True, required_pcc=0.95),
    )

    run_graph_test(
        attention,
        [hidden_states, 0],
        framework=Framework.TORCH,
        comparison_config=comparison_config,
    )


@pytest.mark.nightly
@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("seq_len", [256, 512])
def test_hsa_attention_nightly(batch_size, seq_len, medium_model_args):
    """Nightly test for HSA with larger configurations."""
    xr.set_device_type("TT")

    args = medium_model_args
    args.max_batch_size = batch_size
    args.max_seq_len = seq_len * 2

    attention = HSAAttention(args, layer_id=3)
    attention = attention.to(torch.bfloat16)
    attention.eval()

    hidden_states = torch.randn(
        (batch_size, seq_len, args.dim),
        dtype=torch.bfloat16
    )

    comparison_config = ComparisonConfig(
        pcc=PccConfig(enabled=True, required_pcc=0.95),
    )

    run_graph_test(
        attention,
        [hidden_states, 0],
        framework=Framework.TORCH,
        comparison_config=comparison_config,
    )


# ============================================================================
# Combined CSA + HSA Test (simulating real model layers)
# ============================================================================


@pytest.mark.nightly
@pytest.mark.parametrize("batch_size", [1, 2])
def test_mixed_attention_layers(batch_size, small_model_args):
    """
    Test a sequence of attention layers mixing CSA and HSA.

    This simulates the real DeepSeek V4 architecture where different layers
    use different compression ratios: (0, 0, 4, 128, 4, 128, 4, 0)
    """
    xr.set_device_type("TT")

    args = small_model_args
    args.max_batch_size = batch_size
    seq_len = 128

    # Create a mix of attention layers
    layers = [
        Attention(layer_id=0, args=args),  # compress_ratio=0 (sliding window)
        Attention(layer_id=2, args=args),  # compress_ratio=4 (CSA)
        Attention(layer_id=3, args=args),  # compress_ratio=128 (HSA)
    ]

    for i, layer in enumerate(layers):
        layer = layer.to(torch.bfloat16)
        layer.eval()

        hidden_states = torch.randn(
            (batch_size, seq_len, args.dim),
            dtype=torch.bfloat16
        )

        comparison_config = ComparisonConfig(
            pcc=PccConfig(enabled=True, required_pcc=0.95),
        )

        run_graph_test(
            layer,
            [hidden_states, 0],
            framework=Framework.TORCH,
            comparison_config=comparison_config,
        )
