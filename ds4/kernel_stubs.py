# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Stub implementations for CUDA kernels from DeepSeek V4 Flash.
These are no-op versions for TT-XLA testing in BF16 mode.

The original kernel.py uses custom tilelang kernels that require CUDA.
These stubs allow model.py modules to be tested on TT hardware without
FP8/FP4 quantization (using BF16 throughout instead).

Usage:
    # At the top of test file, before importing from model.py:
    import kernel_stubs
    kernel_stubs.install()

    # Now model.py will use stub implementations
    from model import Compressor, ...
"""
import sys
import torch
from typing import Optional


def act_quant(
    x: torch.Tensor,
    block_size: int = 128,
    scale_fmt: Optional[str] = None,
    scale_dtype: torch.dtype = torch.float32,
    inplace: bool = False,
):
    """
    FP8 activation quantization stub - no-op for BF16 testing.

    In the original: quantizes activations to FP8 with per-block scaling.
    Here: returns input unchanged (BF16 path).
    """
    if inplace:
        return
    return x, None


def fp4_act_quant(x: torch.Tensor, block_size: int = 32, inplace: bool = False):
    """
    FP4 activation quantization stub - no-op for BF16 testing.

    In the original: quantizes activations to FP4 with per-block scaling.
    Here: returns input unchanged (BF16 path).
    """
    if inplace:
        return
    return x, None


def fp8_gemm(
    a: torch.Tensor,
    a_s: torch.Tensor,
    b: torch.Tensor,
    b_s: torch.Tensor,
    scale_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    FP8 GEMM stub - falls back to standard matmul.

    In the original: FP8 matrix multiplication with block scaling.
    Here: standard F.linear (BF16 path).
    """
    return torch.nn.functional.linear(a, b)


def fp4_gemm(
    a: torch.Tensor,
    a_s: torch.Tensor,
    b: torch.Tensor,
    b_s: torch.Tensor,
    scale_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    FP4 GEMM stub - falls back to standard matmul.

    In the original: FP4 matrix multiplication with block scaling.
    Here: standard F.linear (BF16 path).
    """
    return torch.nn.functional.linear(a, b)


def sparse_attn(
    q: torch.Tensor,
    kv: torch.Tensor,
    attn_sink: torch.Tensor,
    topk_idxs: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    """
    Sparse attention via index gathering - accurate CPU implementation.

    Args:
        q: Query tensor [B, S, H, head_dim]
        kv: Key-Value tensor [B, kv_len, head_dim] (shared K=V in MLA)
        attn_sink: Learnable bias per head [H], acts as virtual sink token
        topk_idxs: Indices to gather [B, S, topk], -1 means masked
        softmax_scale: Attention scale factor (typically 1/sqrt(head_dim))

    Returns:
        Output tensor [B, S, H, head_dim]

    Algorithm (matches kernel.py:sparse_attn_kernel):
    1. For each (batch, seq_pos), gather KV at topk_idxs positions
    2. Compute Q @ K^T * scale for gathered positions only
    3. Mask invalid indices (-1) with -inf
    4. Add attn_sink as virtual sink token score
    5. Softmax over gathered positions + sink
    6. Weighted sum of gathered V
    """
    bsz, seqlen, n_heads, head_dim = q.shape
    topk = topk_idxs.shape[-1]
    dtype = q.dtype

    # Work in float32 for numerical stability
    q = q.float()  # [B, S, H, D]
    kv = kv.float()  # [B, kv_len, D]

    # Handle -1 indices by clamping to 0 (will mask scores later)
    gather_idxs = topk_idxs.clamp(min=0).unsqueeze(-1).expand(-1, -1, -1, head_dim)
    # gather_idxs: [B, S, topk, D]

    # Gather KV at topk positions for each seq position
    # kv: [B, kv_len, D] -> expand to [B, S, kv_len, D]
    kv_expanded = kv.unsqueeze(1).expand(-1, seqlen, -1, -1)
    kv_gathered = torch.gather(kv_expanded, dim=2, index=gather_idxs)
    # kv_gathered: [B, S, topk, D]

    # Compute attention scores: Q @ K^T * scale
    # q: [B, S, H, D], kv_gathered: [B, S, topk, D]
    scores = torch.einsum("bshd,bstd->bsht", q, kv_gathered) * softmax_scale
    # scores: [B, S, H, topk]

    # Mask invalid indices (-1) with -inf
    invalid_mask = (topk_idxs == -1).unsqueeze(2)  # [B, S, 1, topk]
    scores = scores.masked_fill(invalid_mask, float("-inf"))

    # Add attn_sink as a virtual sink token
    # The sink absorbs "leftover" attention probability
    sink_scores = attn_sink.view(1, 1, n_heads, 1).expand(bsz, seqlen, -1, 1)
    scores_with_sink = torch.cat([scores, sink_scores], dim=-1)  # [B, S, H, topk+1]

    # Softmax over all positions including sink
    attn_weights = torch.nn.functional.softmax(scores_with_sink, dim=-1)

    # Remove sink weight, keep only topk weights for output computation
    attn_weights = attn_weights[..., :-1]  # [B, S, H, topk]

    # Weighted sum of gathered V (V = K in MLA)
    output = torch.einsum("bsht,bstd->bshd", attn_weights, kv_gathered)
    # output: [B, S, H, D]

    return output.to(dtype)


def hc_split_sinkhorn(
    mixes: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    hc_mult: int = 4,
    sinkhorn_iters: int = 20,
    eps: float = 1e-6,
):
    """
    Pure PyTorch Sinkhorn normalization for Hyper-Connections.

    Args:
        mixes: Raw mixing scores [B, S, (2+hc_mult)*hc_mult] from linear projection
        hc_scale: Learned scaling factors [3] for pre, post, comb
        hc_base: Learned biases [(2+hc_mult)*hc_mult]
        hc_mult: Number of hidden state copies (default 4)
        sinkhorn_iters: Number of Sinkhorn normalization iterations (default 20)
        eps: Small constant for numerical stability

    Returns:
        pre: [B, S, hc_mult] - weights for reducing hc copies to 1
        post: [B, S, hc_mult] - weights for expanding 1 to hc copies
        comb: [B, S, hc_mult, hc_mult] - doubly-stochastic mixing matrix

    Algorithm (matches kernel.py:hc_split_sinkhorn_kernel):
    1. Split mixes into pre, post, comb portions
    2. Apply sigmoid activation with scale and bias
    3. For comb: softmax then iterative Sinkhorn to make doubly-stochastic
    """
    bsz, seqlen, _ = mixes.shape

    # Split and apply activations
    # pre: sigmoid(mixes[0:hc] * scale[0] + base[0:hc]) + eps
    pre = torch.sigmoid(mixes[..., :hc_mult] * hc_scale[0] + hc_base[:hc_mult]) + eps

    # post: 2 * sigmoid(mixes[hc:2*hc] * scale[1] + base[hc:2*hc])
    post = 2 * torch.sigmoid(
        mixes[..., hc_mult:2*hc_mult] * hc_scale[1] + hc_base[hc_mult:2*hc_mult]
    )

    # comb: [B, S, hc_mult, hc_mult]
    comb = mixes[..., 2*hc_mult:].view(bsz, seqlen, hc_mult, hc_mult)
    comb = comb * hc_scale[2] + hc_base[2*hc_mult:].view(hc_mult, hc_mult)

    # Initial softmax on rows + eps
    comb = torch.nn.functional.softmax(comb, dim=-1) + eps

    # Sinkhorn iterations: alternate row and column normalization
    # Makes the matrix approximately doubly-stochastic
    for _ in range(sinkhorn_iters):
        # Row normalization
        comb = comb / (comb.sum(dim=-1, keepdim=True) + eps)
        # Column normalization
        comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)

    return pre, post, comb


def rotate_activation(x: torch.Tensor) -> torch.Tensor:
    """
    Hadamard rotation stub using scipy instead of CUDA fast_hadamard_transform.

    In the original: applies fast Hadamard transform for spreading information
    across dimensions before FP8 quantization.
    Here: uses scipy.linalg.hadamard for CPU testing.
    """
    import scipy.linalg
    assert x.dtype == torch.bfloat16
    hidden_size = x.size(-1)
    # Hadamard matrix with scaling
    hadamard = torch.tensor(
        scipy.linalg.hadamard(hidden_size),
        dtype=torch.bfloat16,
        device=x.device
    ) * (hidden_size ** -0.5)
    return torch.nn.functional.linear(x, hadamard)


# Create a fake module with stub implementations
class _KernelStubModule:
    """Fake kernel module with stub implementations."""
    act_quant = staticmethod(act_quant)
    fp4_act_quant = staticmethod(fp4_act_quant)
    fp8_gemm = staticmethod(fp8_gemm)
    fp4_gemm = staticmethod(fp4_gemm)
    sparse_attn = staticmethod(sparse_attn)
    hc_split_sinkhorn = staticmethod(hc_split_sinkhorn)


# Store original rotate_activation for uninstall
_original_rotate_activation = None


def install():
    """
    Install kernel stubs into sys.modules so model.py imports use stubs.

    Must be called BEFORE importing from model.py.
    """
    sys.modules['kernel'] = _KernelStubModule()

    # Also create a fake fast_hadamard_transform module
    class _FakeHadamardModule:
        @staticmethod
        def hadamard_transform(x, scale=1.0):
            import scipy.linalg
            hidden_size = x.size(-1)
            hadamard = torch.tensor(
                scipy.linalg.hadamard(hidden_size),
                dtype=x.dtype,
                device=x.device
            ) * scale
            return torch.nn.functional.linear(x, hadamard)

    sys.modules['fast_hadamard_transform'] = _FakeHadamardModule()


def uninstall():
    """Remove kernel stubs from sys.modules."""
    if 'kernel' in sys.modules:
        del sys.modules['kernel']
    if 'fast_hadamard_transform' in sys.modules:
        del sys.modules['fast_hadamard_transform']


# ============================================================================
# Simplified Compressor Forward for Static Compilation
# ============================================================================


def compressor_forward_prefill(compressor, x: torch.Tensor):
    """
    Simplified Compressor forward for prefill with seqlen divisible by ratio.
    Statically compilable - no dynamic shapes or conditional branches.

    Assumptions:
    - start_pos == 0 (prefill only)
    - seqlen % compress_ratio == 0 (no remainder)

    This removes:
    - Dynamic shape calculations (remainder, cutoff)
    - Conditional branches (if remainder > 0)
    - In-place state buffer mutations
    - kv_cache write (returns kv directly)
    """
    from model import apply_rotary_emb

    bsz, seqlen, _ = x.size()
    ratio = compressor.compress_ratio
    overlap = compressor.overlap
    rd = compressor.rope_head_dim
    dtype = x.dtype

    # Compression in fp32
    x = x.float()
    kv = compressor.wkv(x)
    score = compressor.wgate(x)

    # Reshape - static shape since seqlen % ratio == 0
    kv = kv.unflatten(1, (-1, ratio))
    score = score.unflatten(1, (-1, ratio)) + compressor.ape

    if overlap:
        kv = compressor.overlap_transform(kv, 0)
        score = compressor.overlap_transform(score, float("-inf"))

    # Gated pooling
    kv = (kv * score.softmax(dim=2)).sum(dim=2)
    kv = compressor.norm(kv.to(dtype))

    # RoPE
    freqs_cis = compressor.freqs_cis[:seqlen:ratio]
    apply_rotary_emb(kv[..., -rd:], freqs_cis)

    # Skip act_quant (stubbed as no-op anyway)
    # Skip kv_cache write (not needed for forward output)

    return kv


def install_compressor_prefill():
    """
    Monkey-patch Compressor.forward with simplified prefill version.

    Must be called AFTER install() and importing model.py.
    """
    from model import Compressor
    if not hasattr(Compressor, '_original_forward'):
        Compressor._original_forward = Compressor.forward
    Compressor.forward = lambda self, x, start_pos: compressor_forward_prefill(self, x)


def uninstall_compressor_prefill():
    """Restore original Compressor.forward."""
    from model import Compressor
    if hasattr(Compressor, '_original_forward'):
        Compressor.forward = Compressor._original_forward
        del Compressor._original_forward


def compressor_forward_decode_accumulate(
    compressor,
    x: torch.Tensor,
    slot_idx: int,
):
    """
    Simplified Compressor forward for decode phase - accumulate only (no compression).
    Statically compilable - no dynamic shapes or conditional branches.

    This is the path where (start_pos + 1) % ratio != 0, so we just insert
    the new token into the state buffer and return None.

    Args:
        compressor: Compressor module instance
        x: Input tensor [B, 1, D] - single decode token
        slot_idx: Pre-computed slot index (start_pos % ratio for non-overlap,
                  ratio + start_pos % ratio for overlap)

    Assumptions:
        - start_pos > 0 (decode phase)
        - (start_pos + 1) % compress_ratio != 0 (not time to compress)
        - seqlen == 1 (single token decode)

    This removes:
        - start_pos % ratio computation (passed as slot_idx)
        - Conditional branches (overlap check, should_compress check)
        - The compression path entirely

    Returns:
        None (no compressed output produced)
    """
    bsz = x.size(0)

    # Compression uses fp32
    x = x.float()
    kv = compressor.wkv(x)      # [B, 1, head_dim] or [B, 1, 2*head_dim] if overlap
    score = compressor.wgate(x)  # same shape

    # Add positional encoding for current position within compression window
    # Note: slot_idx % ratio gives position within window
    pos_in_window = slot_idx % compressor.compress_ratio
    score = score + compressor.ape[pos_in_window]

    # Insert into state buffer at the designated slot
    # For overlap=True: slot_idx = ratio + (start_pos % ratio), so range [ratio, 2*ratio)
    # For overlap=False: slot_idx = start_pos % ratio, so range [0, ratio)
    compressor.kv_state[:bsz, slot_idx] = kv.squeeze(1)
    compressor.score_state[:bsz, slot_idx] = score.squeeze(1)

    # No compression happens, no output
    return None


def compressor_forward_decode_accumulate_overlap(
    compressor,
    x: torch.Tensor,
    start_pos: int,
):
    """
    Decode accumulate for CSA (overlap=True, ratio=4).

    Convenience wrapper that computes slot_idx for overlap case.
    """
    slot_idx = compressor.compress_ratio + (start_pos % compressor.compress_ratio)
    return compressor_forward_decode_accumulate(compressor, x, slot_idx)


def compressor_forward_decode_accumulate_no_overlap(
    compressor,
    x: torch.Tensor,
    start_pos: int,
):
    """
    Decode accumulate for HSA (overlap=False, ratio=128).

    Convenience wrapper that computes slot_idx for non-overlap case.
    """
    slot_idx = start_pos % compressor.compress_ratio
    return compressor_forward_decode_accumulate(compressor, x, slot_idx)


# ============================================================================
# RatioPadCompressor: Compressor variant with padding for static shapes
# ============================================================================


def ratio_pad_compressor_forward_prefill(
    compressor,
    x: torch.Tensor,
    kv: torch.Tensor,
    score: torch.Tensor,
):
    """
    Compressor forward for prefill that pads inputs to a multiple of compress_ratio.
    This eliminates dynamic shapes in unflatten operations for better compiler compatibility.

    Args:
        compressor: Compressor module instance (provides weights, buffers, config)
        x: Original input tensor [B, seqlen, D] - needed for state buffer updates
        kv: Pre-computed kv = compressor.wkv(x.float()) [B, seqlen, head_dim or 2*head_dim]
        score: Pre-computed score = compressor.wgate(x.float()) [B, seqlen, head_dim or 2*head_dim]

    Returns:
        kv: Compressed KV tensor [B, seqlen // ratio, head_dim]

    Key difference from original Compressor.forward:
    - Pads kv/score to multiple of ratio before unflatten
    - Uses explicit (num_compressed, ratio) shape instead of (-1, ratio)
    - Masks padded positions with -inf scores so they contribute 0 after softmax
    - Trims output to valid_compressed positions

    Note: This function focuses on the core compression logic. RoPE, quantization,
    and kv_cache writes should be handled by the caller if needed.
    """
    from model import apply_rotary_emb

    bsz, seqlen, _ = x.size()
    ratio = compressor.compress_ratio
    overlap = compressor.overlap
    d = compressor.head_dim
    rd = compressor.rope_head_dim
    dtype = x.dtype

    # Calculate padding
    remainder = seqlen % ratio
    if remainder > 0:
        pad_len = ratio - remainder
        # Pad kv with zeros (will be masked out by -inf scores)
        kv = torch.nn.functional.pad(kv, (0, 0, 0, pad_len), value=0)
        # Pad score with -inf so padded positions contribute zero weight after softmax
        score = torch.nn.functional.pad(score, (0, 0, 0, pad_len), value=float("-inf"))
        padded_seqlen = seqlen + pad_len
    else:
        padded_seqlen = seqlen

    num_compressed = padded_seqlen // ratio

    # Store state for overlap (uses original unpadded positions)
    offset = ratio if overlap else 0
    if overlap and seqlen >= ratio:
        # Use last complete window from original sequence for overlap state
        cutoff = seqlen - remainder if remainder > 0 else seqlen
        if cutoff >= ratio:
            compressor.kv_state[:bsz, :ratio] = compressor.wkv(x[:, cutoff-ratio:cutoff].float())
            compressor.score_state[:bsz, :ratio] = compressor.wgate(x[:, cutoff-ratio:cutoff].float()) + compressor.ape

    # Store remainder tokens in state buffer (original remainder, not padding)
    if remainder > 0:
        compressor.kv_state[:bsz, offset:offset+remainder] = compressor.wkv(x[:, seqlen-remainder:].float())
        compressor.score_state[:bsz, offset:offset+remainder] = compressor.wgate(x[:, seqlen-remainder:].float()) + compressor.ape[:remainder]

    # Static shape unflatten: padded_seqlen is guaranteed to be multiple of ratio
    kv = kv.unflatten(1, (num_compressed, ratio))
    score = score.unflatten(1, (num_compressed, ratio)) + compressor.ape

    if overlap:
        kv = compressor.overlap_transform(kv, 0)
        score = compressor.overlap_transform(score, float("-inf"))

    # Weighted sum with softmax (padded positions have -inf scores, contribute 0)
    kv = (kv * score.softmax(dim=2)).sum(dim=2)

    # Only keep valid compressed positions (exclude those from padding)
    valid_compressed = seqlen // ratio
    if valid_compressed < num_compressed:
        kv = kv[:, :valid_compressed]

    # Normalization
    kv = compressor.norm(kv.to(dtype))

    # RoPE on last rd dimensions
    freqs_cis = compressor.freqs_cis[:valid_compressed * ratio:ratio]
    apply_rotary_emb(kv[..., -rd:], freqs_cis)

    # Skip act_quant (stubbed as no-op anyway in BF16 mode)

    # Write to kv_cache
    if compressor.kv_cache is not None:
        compressor.kv_cache[:bsz, :valid_compressed] = kv

    return kv


def ratio_pad_compressor_forward(compressor, x: torch.Tensor, start_pos: int):
    """
    Full Compressor forward replacement that uses padding for static shapes.

    This is a drop-in replacement for Compressor.forward that:
    - Uses padding in prefill for static unflatten shapes
    - Falls back to original decode logic (single token, no padding needed)

    Usage:
        # After install() and importing model:
        from model import Compressor
        compressor = Compressor(args, compress_ratio=4)
        # ... setup kv_cache, freqs_cis ...

        # Use padded forward instead of compressor.forward(x, start_pos)
        kv = ratio_pad_compressor_forward(compressor, x, start_pos=0)
    """
    bsz, seqlen, _ = x.size()
    ratio = compressor.compress_ratio
    overlap = compressor.overlap
    d = compressor.head_dim
    rd = compressor.rope_head_dim
    dtype = x.dtype

    # Compression uses fp32
    x_float = x.float()
    kv = compressor.wkv(x_float)
    score = compressor.wgate(x_float)

    if start_pos == 0:
        # Prefill: use padded version
        return ratio_pad_compressor_forward_prefill(compressor, x, kv, score)
    else:
        # Decode: original logic (single token, no padding needed)
        from model import apply_rotary_emb

        should_compress = (start_pos + 1) % ratio == 0
        score = score + compressor.ape[start_pos % ratio]

        if overlap:
            compressor.kv_state[:bsz, ratio + start_pos % ratio] = kv.squeeze(1)
            compressor.score_state[:bsz, ratio + start_pos % ratio] = score.squeeze(1)
            if should_compress:
                kv_state = torch.cat([compressor.kv_state[:bsz, :ratio, :d], compressor.kv_state[:bsz, ratio:, d:]], dim=1)
                score_state = torch.cat([compressor.score_state[:bsz, :ratio, :d], compressor.score_state[:bsz, ratio:, d:]], dim=1)
                kv = (kv_state * score_state.softmax(dim=1)).sum(dim=1, keepdim=True)
                compressor.kv_state[:bsz, :ratio] = compressor.kv_state[:bsz, ratio:]
                compressor.score_state[:bsz, :ratio] = compressor.score_state[:bsz, ratio:]
        else:
            compressor.kv_state[:bsz, start_pos % ratio] = kv.squeeze(1)
            compressor.score_state[:bsz, start_pos % ratio] = score.squeeze(1)
            if should_compress:
                kv = (compressor.kv_state[:bsz] * compressor.score_state[:bsz].softmax(dim=1)).sum(dim=1, keepdim=True)

        if not should_compress:
            return None

        kv = compressor.norm(kv.to(dtype))
        freqs_cis = compressor.freqs_cis[start_pos + 1 - ratio].unsqueeze(0)
        apply_rotary_emb(kv[..., -rd:], freqs_cis)

        # Skip act_quant (stubbed as no-op)

        if compressor.kv_cache is not None:
            compressor.kv_cache[:bsz, start_pos // ratio] = kv.squeeze(1)

        return kv


def install_ratio_pad_compressor():
    """
    Monkey-patch Compressor.forward with ratio-padded version.

    Must be called AFTER install() and importing model.py.
    """
    from model import Compressor
    if not hasattr(Compressor, '_original_forward'):
        Compressor._original_forward = Compressor.forward
    Compressor.forward = lambda self, x, start_pos: ratio_pad_compressor_forward(self, x, start_pos)


def uninstall_ratio_pad_compressor():
    """Restore original Compressor.forward."""
    from model import Compressor
    if hasattr(Compressor, '_original_forward'):
        Compressor.forward = Compressor._original_forward
        del Compressor._original_forward
