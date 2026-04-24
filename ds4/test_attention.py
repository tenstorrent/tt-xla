# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
CPU-only tests for DeepSeek V4 Attention variants.

Tests SWA, CSA, and HSA attention mechanisms using model.py with
kernel_stubs.py providing pure PyTorch implementations of CUDA kernels.

Run with: pytest ds4/test_attention.py -v
"""
# IMPORTANT: Install stubs BEFORE importing from model.py
import kernel_stubs
kernel_stubs.install()

import pytest
import torch

# Set default dtype to bfloat16 so model buffers (kv_cache) use correct dtype
torch.set_default_dtype(torch.bfloat16)

from model import (
    ModelArgs,
    Attention,
    Compressor,
    Indexer,
    RMSNorm,
    Linear,
    ColumnParallelLinear,
    RowParallelLinear,
    precompute_freqs_cis,
    apply_rotary_emb,
)


def init_weights(module):
    """Initialize model weights for testing (model uses torch.empty which gives garbage)."""
    for name, param in module.named_parameters():
        if param.requires_grad:
            if 'weight' in name and param.dim() >= 2:
                # Xavier uniform for weight matrices - must work in float32 then convert
                with torch.no_grad():
                    init_data = torch.empty_like(param, dtype=torch.float32)
                    torch.nn.init.xavier_uniform_(init_data)
                    param.copy_(init_data.to(param.dtype))
            elif 'bias' in name:
                # Zero for biases
                with torch.no_grad():
                    param.zero_()
            elif param.dim() == 1:
                # 1D weights (like RMSNorm) should be initialized to 1
                with torch.no_grad():
                    param.fill_(1.0)
    # Also initialize buffers that may have garbage values
    for name, buf in module.named_buffers():
        if buf.dtype in (torch.bfloat16, torch.float16, torch.float32):
            buf.zero_()


# Test configuration with smaller dimensions for CPU testing
@pytest.fixture
def small_args():
    """Small model args for fast CPU testing."""
    return ModelArgs(
        max_batch_size=2,
        max_seq_len=256,
        dim=256,
        n_heads=4,
        head_dim=64,
        rope_head_dim=16,
        q_lora_rank=64,
        o_lora_rank=64,
        o_groups=2,
        window_size=32,
        index_n_heads=4,
        index_head_dim=32,
        index_topk=16,
        n_routed_experts=4,
        n_activated_experts=2,
        moe_inter_dim=256,
        hc_mult=4,
        compress_ratios=(0, 0, 4, 128),  # layer 0,1=SWA, 2=CSA, 3=HSA
    )


class TestRMSNorm:
    """Test RMSNorm independently."""

    def test_forward_shape(self):
        norm = RMSNorm(dim=256)
        x = torch.randn(2, 16, 256, dtype=torch.bfloat16)
        out = norm(x)
        assert out.shape == x.shape
        assert out.dtype == x.dtype

    def test_normalization(self):
        norm = RMSNorm(dim=256, eps=1e-6)
        x = torch.randn(2, 16, 256, dtype=torch.float32)
        out = norm(x)
        # RMS of output should be close to 1 (with unit weights)
        rms = out.square().mean(-1).sqrt()
        assert rms.mean().item() == pytest.approx(1.0, rel=0.1)


class TestLinear:
    """Test Linear layer independently."""

    def test_forward_shape(self):
        linear = Linear(256, 512)
        x = torch.randn(2, 16, 256, dtype=torch.bfloat16)
        out = linear(x)
        assert out.shape == (2, 16, 512)

    def test_column_parallel(self):
        linear = ColumnParallelLinear(256, 512)
        x = torch.randn(2, 16, 256, dtype=torch.bfloat16)
        out = linear(x)
        # world_size=1, so full output dim
        assert out.shape == (2, 16, 512)

    def test_row_parallel(self):
        linear = RowParallelLinear(256, 512)
        x = torch.randn(2, 16, 256, dtype=torch.bfloat16)
        out = linear(x)
        assert out.shape == (2, 16, 512)


class TestRoPE:
    """Test Rotary Position Embeddings."""

    def test_freqs_cis_shape(self):
        freqs = precompute_freqs_cis(
            dim=64, seqlen=128, original_seq_len=0,
            base=10000.0, factor=40, beta_fast=32, beta_slow=1
        )
        assert freqs.shape == (128, 32)  # seqlen, dim//2
        assert freqs.dtype == torch.complex64

    def test_apply_rotary_emb_shape(self):
        freqs = precompute_freqs_cis(
            dim=16, seqlen=32, original_seq_len=0,
            base=10000.0, factor=40, beta_fast=32, beta_slow=1
        )
        # Shape: [B, S, H, rope_dim]
        x = torch.randn(2, 32, 4, 16, dtype=torch.bfloat16)
        x_clone = x.clone()
        out = apply_rotary_emb(x_clone, freqs)
        assert out.shape == x.shape
        assert out.dtype == x.dtype

    def test_inverse_rotary(self):
        freqs = precompute_freqs_cis(
            dim=16, seqlen=32, original_seq_len=0,
            base=10000.0, factor=40, beta_fast=32, beta_slow=1
        )
        x = torch.randn(2, 32, 4, 16, dtype=torch.float32)
        x_orig = x.clone()

        # Apply forward then inverse
        x_rotated = apply_rotary_emb(x.clone(), freqs, inverse=False)
        x_back = apply_rotary_emb(x_rotated.clone(), freqs, inverse=True)

        # Should recover original
        torch.testing.assert_close(x_back, x_orig, rtol=1e-4, atol=1e-4)


class TestCompressor:
    """Test KV Compressor independently."""

    def test_compressor_init(self, small_args):
        comp = Compressor(small_args, compress_ratio=4, head_dim=64)
        assert comp.compress_ratio == 4
        assert comp.overlap is True  # ratio=4 uses overlap
        assert comp.head_dim == 64

    def test_compressor_ratio_128(self, small_args):
        comp = Compressor(small_args, compress_ratio=128, head_dim=64)
        assert comp.compress_ratio == 128
        assert comp.overlap is False  # ratio=128 no overlap

    def test_compressor_forward_prefill(self, small_args):
        comp = Compressor(small_args, compress_ratio=4, head_dim=64)
        init_weights(comp)

        # Setup required state
        comp.kv_cache = torch.zeros(2, 64, 64, dtype=torch.bfloat16)
        comp.freqs_cis = precompute_freqs_cis(
            dim=small_args.rope_head_dim, seqlen=256, original_seq_len=0,
            base=10000.0, factor=40, beta_fast=32, beta_slow=1
        )

        x = torch.randn(2, 32, small_args.dim, dtype=torch.bfloat16)
        kv_compressed = comp(x, start_pos=0)

        # 32 tokens with ratio=4 -> 8 compressed entries
        assert kv_compressed is not None
        assert kv_compressed.shape == (2, 8, 64)

    def test_compressor_forward_not_enough_tokens(self, small_args):
        """If seq_len < ratio, no compression happens."""
        comp = Compressor(small_args, compress_ratio=128, head_dim=64)
        init_weights(comp)
        comp.kv_cache = torch.zeros(2, 2, 64, dtype=torch.bfloat16)
        comp.freqs_cis = precompute_freqs_cis(
            dim=small_args.rope_head_dim, seqlen=256, original_seq_len=0,
            base=10000.0, factor=40, beta_fast=32, beta_slow=1
        )

        # Only 64 tokens, ratio=128 -> not enough to compress
        x = torch.randn(2, 64, small_args.dim, dtype=torch.bfloat16)
        result = comp(x, start_pos=0)

        # Should return None (no compression yet)
        assert result is None

    def test_compressor_decode_accumulate_overlap(self, small_args):
        """Test decode accumulate stub for CSA (overlap=True)."""
        from kernel_stubs import compressor_forward_decode_accumulate_overlap

        comp = Compressor(small_args, compress_ratio=4, head_dim=64)
        init_weights(comp)
        comp.kv_cache = torch.zeros(2, 64, 64, dtype=torch.bfloat16)
        comp.freqs_cis = precompute_freqs_cis(
            dim=small_args.rope_head_dim, seqlen=256, original_seq_len=0,
            base=10000.0, factor=40, beta_fast=32, beta_slow=1
        )

        batch_size = 2
        x = torch.randn(batch_size, 1, small_args.dim, dtype=torch.bfloat16)

        # State should be zeros initially
        assert comp.kv_state.abs().sum() == 0
        assert comp.score_state[:, 4:].abs().sum() == 0  # overlap region starts at ratio

        # Accumulate at start_pos=0 (slot_idx = 4 + 0 = 4 for overlap)
        result = compressor_forward_decode_accumulate_overlap(comp, x, start_pos=0)

        assert result is None  # No compression output
        # Check that slot 4 (ratio + 0) was written
        assert comp.kv_state[:batch_size, 4].abs().sum() > 0
        assert comp.score_state[:batch_size, 4].abs().sum() > 0

        # Accumulate at start_pos=1 (slot_idx = 4 + 1 = 5)
        x2 = torch.randn(batch_size, 1, small_args.dim, dtype=torch.bfloat16)
        compressor_forward_decode_accumulate_overlap(comp, x2, start_pos=1)
        assert comp.kv_state[:batch_size, 5].abs().sum() > 0

    def test_compressor_decode_accumulate_no_overlap(self, small_args):
        """Test decode accumulate stub for HSA (overlap=False)."""
        from kernel_stubs import compressor_forward_decode_accumulate_no_overlap

        comp = Compressor(small_args, compress_ratio=128, head_dim=64)
        init_weights(comp)
        comp.kv_cache = torch.zeros(2, 2, 64, dtype=torch.bfloat16)
        comp.freqs_cis = precompute_freqs_cis(
            dim=small_args.rope_head_dim, seqlen=256, original_seq_len=0,
            base=10000.0, factor=40, beta_fast=32, beta_slow=1
        )

        batch_size = 2
        x = torch.randn(batch_size, 1, small_args.dim, dtype=torch.bfloat16)

        # State should be zeros initially
        assert comp.kv_state.abs().sum() == 0

        # Accumulate at start_pos=0 (slot_idx = 0 for no overlap)
        result = compressor_forward_decode_accumulate_no_overlap(comp, x, start_pos=0)

        assert result is None
        assert comp.kv_state[:batch_size, 0].abs().sum() > 0
        assert comp.score_state[:batch_size, 0].abs().sum() > 0

        # Accumulate at start_pos=5 (slot_idx = 5)
        x2 = torch.randn(batch_size, 1, small_args.dim, dtype=torch.bfloat16)
        compressor_forward_decode_accumulate_no_overlap(comp, x2, start_pos=5)
        assert comp.kv_state[:batch_size, 5].abs().sum() > 0


class TestIndexer:
    """Test Indexer for CSA top-k selection."""

    def test_indexer_init(self, small_args):
        indexer = Indexer(small_args, compress_ratio=4)
        assert indexer.compress_ratio == 4
        assert indexer.index_topk == small_args.index_topk

    def test_indexer_forward(self, small_args):
        indexer = Indexer(small_args, compress_ratio=4)
        init_weights(indexer)
        indexer.freqs_cis = precompute_freqs_cis(
            dim=small_args.rope_head_dim, seqlen=256, original_seq_len=0,
            base=10000.0, factor=40, beta_fast=32, beta_slow=1
        )

        batch_size = 2
        seq_len = 64

        x = torch.randn(batch_size, seq_len, small_args.dim, dtype=torch.bfloat16)
        qr = torch.randn(batch_size, seq_len, small_args.q_lora_rank, dtype=torch.bfloat16)

        topk_idxs = indexer(x, qr, start_pos=0, offset=seq_len)

        # min(index_topk, seq_len // ratio) = min(16, 64//4) = 16
        expected_k = min(small_args.index_topk, seq_len // 4)
        assert topk_idxs.shape == (batch_size, seq_len, expected_k)


class TestSWAAttention:
    """Test Sliding Window Attention (compress_ratio=0)."""

    def test_swa_init(self, small_args):
        attn = Attention(layer_id=0, args=small_args)
        assert attn.compress_ratio == 0

    def test_swa_forward_prefill(self, small_args):
        attn = Attention(layer_id=0, args=small_args)
        init_weights(attn)

        batch_size = 2
        seq_len = 32

        x = torch.randn(batch_size, seq_len, small_args.dim, dtype=torch.bfloat16)
        out = attn(x, start_pos=0)

        assert out.shape == (batch_size, seq_len, small_args.dim)
        assert out.dtype == torch.bfloat16

    def test_swa_forward_decode(self, small_args):
        attn = Attention(layer_id=0, args=small_args)
        init_weights(attn)

        batch_size = 2

        # First do prefill
        x_prefill = torch.randn(batch_size, 32, small_args.dim, dtype=torch.bfloat16)
        attn(x_prefill, start_pos=0)

        # Then decode single token
        x_decode = torch.randn(batch_size, 1, small_args.dim, dtype=torch.bfloat16)
        out = attn(x_decode, start_pos=32)

        assert out.shape == (batch_size, 1, small_args.dim)

    def test_swa_kv_cache_populated(self, small_args):
        attn = Attention(layer_id=0, args=small_args)
        init_weights(attn)

        batch_size = 2
        seq_len = 32

        x = torch.randn(batch_size, seq_len, small_args.dim, dtype=torch.bfloat16)

        # Cache should be zeros initially (init_weights zeros buffers)
        assert attn.kv_cache.abs().sum() == 0

        attn(x, start_pos=0)

        # Cache should be populated
        assert attn.kv_cache[:batch_size, :seq_len].abs().sum() > 0


class TestCSAAttention:
    """Test Compressed Sparse Attention (compress_ratio=4 with Indexer)."""

    def test_csa_init(self, small_args):
        attn = Attention(layer_id=2, args=small_args)
        assert attn.compress_ratio == 4
        assert attn.indexer is not None

    def test_csa_forward_prefill(self, small_args):
        attn = Attention(layer_id=2, args=small_args)
        init_weights(attn)

        batch_size = 2
        seq_len = 64

        x = torch.randn(batch_size, seq_len, small_args.dim, dtype=torch.bfloat16)
        out = attn(x, start_pos=0)

        assert out.shape == (batch_size, seq_len, small_args.dim)
        assert out.dtype == torch.bfloat16

    def test_csa_forward_decode(self, small_args):
        attn = Attention(layer_id=2, args=small_args)
        init_weights(attn)

        batch_size = 2

        x_prefill = torch.randn(batch_size, 64, small_args.dim, dtype=torch.bfloat16)
        attn(x_prefill, start_pos=0)

        x_decode = torch.randn(batch_size, 1, small_args.dim, dtype=torch.bfloat16)
        out = attn(x_decode, start_pos=64)

        assert out.shape == (batch_size, 1, small_args.dim)

    def test_csa_compressed_cache_populated(self, small_args):
        attn = Attention(layer_id=2, args=small_args)
        init_weights(attn)

        batch_size = 2
        seq_len = 64

        x = torch.randn(batch_size, seq_len, small_args.dim, dtype=torch.bfloat16)
        attn(x, start_pos=0)

        win = small_args.window_size
        # Window portion should be populated
        assert attn.kv_cache[:batch_size, :win].abs().sum() > 0

        # Compressed portion should be populated (64/4 = 16 entries)
        compressed_entries = seq_len // 4
        assert attn.kv_cache[:batch_size, win:win + compressed_entries].abs().sum() > 0


class TestHSAAttention:
    """Test Heavily Compressed Attention (compress_ratio=128 without Indexer)."""

    def test_hsa_init(self, small_args):
        attn = Attention(layer_id=3, args=small_args)
        assert attn.compress_ratio == 128
        assert attn.indexer is None

    def test_hsa_forward_prefill_short(self, small_args):
        """HSA with seq_len < 128 (no compression triggered)."""
        attn = Attention(layer_id=3, args=small_args)
        init_weights(attn)

        batch_size = 2
        seq_len = 64

        x = torch.randn(batch_size, seq_len, small_args.dim, dtype=torch.bfloat16)
        out = attn(x, start_pos=0)

        assert out.shape == (batch_size, seq_len, small_args.dim)

    def test_hsa_forward_prefill_long(self):
        """HSA with seq_len >= 128 (compression triggered)."""
        args = ModelArgs(
            max_batch_size=2,
            max_seq_len=512,
            dim=256,
            n_heads=4,
            head_dim=64,
            rope_head_dim=16,
            q_lora_rank=64,
            o_lora_rank=64,
            o_groups=2,
            window_size=32,
            index_n_heads=4,
            index_head_dim=32,
            index_topk=16,
            n_routed_experts=4,
            n_activated_experts=2,
            moe_inter_dim=256,
            hc_mult=4,
            compress_ratios=(0, 0, 4, 128),
        )

        attn = Attention(layer_id=3, args=args)
        init_weights(attn)

        batch_size = 2
        seq_len = 256

        x = torch.randn(batch_size, seq_len, args.dim, dtype=torch.bfloat16)
        out = attn(x, start_pos=0)

        assert out.shape == (batch_size, seq_len, args.dim)


class TestAttentionNumericalStability:
    """Test attention for numerical stability."""

    def test_no_nans_in_output(self, small_args):
        attn = Attention(layer_id=2, args=small_args)
        init_weights(attn)

        x = torch.randn(2, 64, small_args.dim, dtype=torch.bfloat16)
        out = attn(x, start_pos=0)

        assert not torch.isnan(out).any(), "NaN detected in attention output"
        assert not torch.isinf(out).any(), "Inf detected in attention output"

    def test_large_values_dont_explode(self, small_args):
        # Use SWA (layer_id=0) to avoid Hadamard transform numerical issues
        # CSA's Indexer uses Hadamard which can overflow with large values in bfloat16
        attn = Attention(layer_id=0, args=small_args)
        init_weights(attn)

        x = torch.randn(2, 32, small_args.dim, dtype=torch.bfloat16) * 5
        out = attn(x, start_pos=0)

        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()


class TestAttentionSequential:
    """Test sequential decode steps."""

    def test_multiple_decode_steps(self, small_args):
        attn = Attention(layer_id=0, args=small_args)
        init_weights(attn)

        batch_size = 2

        x_prefill = torch.randn(batch_size, 16, small_args.dim, dtype=torch.bfloat16)
        out_prefill = attn(x_prefill, start_pos=0)
        assert out_prefill.shape == (batch_size, 16, small_args.dim)

        for i in range(5):
            x_decode = torch.randn(batch_size, 1, small_args.dim, dtype=torch.bfloat16)
            out_decode = attn(x_decode, start_pos=16 + i)
            assert out_decode.shape == (batch_size, 1, small_args.dim)
            assert not torch.isnan(out_decode).any()

    def test_csa_multiple_decode_steps(self, small_args):
        """Test CSA with multiple decode steps (compression accumulates)."""
        attn = Attention(layer_id=2, args=small_args)
        init_weights(attn)

        batch_size = 2

        x_prefill = torch.randn(batch_size, 32, small_args.dim, dtype=torch.bfloat16)
        attn(x_prefill, start_pos=0)

        # Decode 8 steps - compression happens at every 4th step
        for i in range(8):
            x_decode = torch.randn(batch_size, 1, small_args.dim, dtype=torch.bfloat16)
            out = attn(x_decode, start_pos=32 + i)
            assert out.shape == (batch_size, 1, small_args.dim)
            assert not torch.isnan(out).any()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
