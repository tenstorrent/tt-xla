# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Block-wise FP8 dequantization.

Currently covers DeepSeek-style block-wise FP8 (one `weight_scale_inv` scalar
per 128x128 block, stored alongside each weight as `<key>.weight_scale_inv`).
Other quant schemes (per-row scale, per-tensor scale, FP4, GPTQ groups) will
need separate helpers — keep them in this module so the spec API can stay
dtype-driven.
"""
import torch

FP8_BLOCK_SIZE = 128

# Treat all native FP8 dtypes as dequantizable. Add new ones as torch ships them.
_FP8_DTYPES = (torch.float8_e4m3fn, torch.float8_e5m2)


def is_fp8(tensor: torch.Tensor) -> bool:
    """True iff `tensor` has a native FP8 dtype."""
    return tensor.dtype in _FP8_DTYPES


def fp8_blockwise_dequant(
    weight: torch.Tensor,
    scale_inv: torch.Tensor,
    block_size: int = FP8_BLOCK_SIZE,
) -> torch.Tensor:
    """Dequantize a 2D FP8 weight to BF16 using block-wise scales.

    `weight` is FP8 with shape `[rows, cols]`; `scale_inv` is float with shape
    `[ceil(rows/block_size), ceil(cols/block_size)]`. Each `block_size x
    block_size` tile of `weight` is multiplied by its scalar `scale_inv` entry.

    Padding: if rows or cols aren't multiples of `block_size`, the weight is
    zero-padded to the next multiple, dequantized, then sliced back to the
    original shape.
    """
    assert weight.dim() == 2, f"expected 2D FP8 weight, got shape {tuple(weight.shape)}"
    rows, cols = weight.shape
    pad_rows = (block_size - rows % block_size) % block_size
    pad_cols = (block_size - cols % block_size) % block_size
    if pad_rows or pad_cols:
        weight = torch.nn.functional.pad(weight, (0, pad_cols, 0, pad_rows))
    padded_rows, padded_cols = weight.shape
    n_br = padded_rows // block_size
    n_bc = padded_cols // block_size

    # Reshape into (n_br, n_bc, block, block) so each block is contiguous,
    # multiply by the corresponding scalar scale, then reshape back.
    weight = (
        weight.view(n_br, block_size, n_bc, block_size)
        .transpose(1, 2)
        .contiguous()
        .view(-1, block_size * block_size)
    )
    weight = (
        (weight.float() * scale_inv.view(-1, 1).float())
        .to(torch.bfloat16)
        .view(n_br, n_bc, block_size, block_size)
        .transpose(1, 2)
        .contiguous()
        .view(padded_rows, padded_cols)
    )
    return weight[:rows, :cols]


def maybe_dequant(tensor: torch.Tensor, scale: torch.Tensor | None) -> torch.Tensor:
    """Dequantize `tensor` iff it's FP8 and `scale` is provided.

    Otherwise return `tensor` unchanged. Lets `transform_group` closures stay
    dtype-agnostic for BF16-source models like GLM.
    """
    if is_fp8(tensor) and scale is not None:
        return fp8_blockwise_dequant(tensor, scale)
    return tensor
