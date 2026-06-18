# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Load-time dequantization of DeepSeek block-wise FP8 weights.

DeepSeek-V3.x ships its linear weights in block-wise FP8: each quantized linear
stores a ``…weight`` tensor in ``float8_e4m3fn`` plus a companion
``…weight_scale_inv`` tensor holding one ``float32`` scale per ``block_size ×
block_size`` block of the weight. The real weight is recovered by multiplying
each block of the FP8 values by its scale (see the reference ``weight_dequant``
in ``tt_forge_models/deepseek/deepseek_v3_2_exp/.../original_model.py``):

    W_real[i, j] = fp8_weight[i, j] * scale[i // block, j // block]

There is no FP8 matmul anywhere in the TT stack (see ``FP8_INDEXER_GAP.md``), so
to run a real DeepSeek checkpoint the weights must be dequantized to bf16 before
compile. The model is built with ``quantization_config=None`` (plain bf16
linears, which dodges vLLM's ``Fp8LinearMethod`` ``KeyError`` on the OOT
platform); without this transform two things break in
``DeepseekV2ForCausalLM.load_weights``:

  1. the FP8 ``…weight`` is cast to bf16 element-wise but its per-block scale is
     never applied, so every block is off by its (power-of-two) scale factor;
  2. the orphan ``…weight_scale_inv`` tensors have no destination parameter and
     hit ``params_dict[name]`` → ``KeyError``, crashing the load.

This module pairs each FP8 weight with its scale, applies the block dequant, and
yields a bf16 weight under the original ``…weight`` name while consuming (not
yielding) the scale tensors — so the downstream loader sees a plain bf16
checkpoint. Optionally pair with ``experimental_weight_dtype="bfp_bf8"`` to then
store the result on-device as 8-bit block-float.
"""
from typing import Iterable, Iterator, Tuple

import torch

from .logger import tt_init_logger

logger = tt_init_logger(__name__)

# Companion scale tensor suffix used by DeepSeek block-wise FP8 checkpoints. The
# scale for ``<name>.weight`` is stored as ``<name>.weight_scale_inv``.
_SCALE_SUFFIX = "_scale_inv"

# 1-byte IEEE FP8 dtypes that carry block scales. Built defensively since not
# every variant exists on every torch build.
_FP8_DTYPES = tuple(
    dt
    for dt in (
        getattr(torch, "float8_e4m3fn", None),
        getattr(torch, "float8_e4m3fnuz", None),
        getattr(torch, "float8_e5m2", None),
        getattr(torch, "float8_e5m2fnuz", None),
    )
    if dt is not None
)

WeightItem = Tuple[str, torch.Tensor]


def _ceil_div(a: int, b: int) -> int:
    return -(-a // b)


def block_dequant(
    weight: torch.Tensor,
    scale: torch.Tensor,
    *,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Dequantize a block-wise FP8 ``weight`` using its per-block ``scale``.

    ``weight`` is ``[out, in]`` in an FP8 dtype; ``scale`` is
    ``[ceil(out / block), ceil(in / block)]`` of per-block multipliers. The
    block size is inferred from the shapes (DeepSeek uses 128×128) so no config
    is needed, and ragged trailing blocks are handled by slicing.
    """
    w = weight.to(torch.float32)
    s = scale.to(torch.float32)
    out_dim, in_dim = w.shape
    scale_rows, scale_cols = s.shape
    # Expand each scale entry across its block and crop to the weight shape.
    # repeat factor == block size when the dim divides evenly (the common case).
    s = s.repeat_interleave(_ceil_div(out_dim, scale_rows), dim=0)
    s = s.repeat_interleave(_ceil_div(in_dim, scale_cols), dim=1)
    s = s[:out_dim, :in_dim]
    return (w * s).to(out_dtype)


def dequantize_block_fp8_weights(
    weights: Iterable[WeightItem],
) -> Iterator[WeightItem]:
    """Stream-transform a checkpoint weight iterator, dequantizing block FP8.

    For every ``(name, tensor)`` from ``weights``:
      - an FP8 ``…weight`` is paired with its ``…weight_scale_inv`` scale,
        dequantized to bf16, and yielded under the original name;
      - the ``…weight_scale_inv`` scale tensor is consumed and never yielded;
      - all other tensors pass through unchanged.

    Weight and scale may arrive in either order, so each is buffered until its
    partner is seen. Buffering is bounded by the number of in-flight unmatched
    tensors (typically tiny — a weight and its scale sit adjacent in a shard),
    not the whole checkpoint, so this stays streaming-friendly for large models.
    """
    pending_weight: dict[str, torch.Tensor] = {}  # weight name -> FP8 tensor
    pending_scale: dict[str, torch.Tensor] = {}  # weight name -> scale tensor
    dequantized = 0

    for name, tensor in weights:
        if name.endswith(_SCALE_SUFFIX):
            weight_name = name[: -len(_SCALE_SUFFIX)]
            weight = pending_weight.pop(weight_name, None)
            if weight is not None:
                yield weight_name, block_dequant(weight, tensor)
                dequantized += 1
            else:
                pending_scale[weight_name] = tensor
        elif tensor.dtype in _FP8_DTYPES:
            scale = pending_scale.pop(name, None)
            if scale is not None:
                yield name, block_dequant(tensor, scale)
                dequantized += 1
            else:
                pending_weight[name] = tensor
        else:
            yield name, tensor

    # Leftovers indicate a checkpoint that doesn't match the expected
    # weight/scale pairing. Surface it loudly but degrade gracefully.
    for name, weight in pending_weight.items():
        logger.warning(
            "Block-FP8 weight %r had no matching %r; casting to bf16 without a "
            "scale (numerics will be wrong).",
            name,
            name + _SCALE_SUFFIX,
        )
        yield name, weight.to(torch.bfloat16)
    for name in pending_scale:
        logger.warning(
            "Block-FP8 scale %r had no matching FP8 weight; dropping it.",
            name + _SCALE_SUFFIX,
        )

    logger.info("Dequantized %d block-FP8 weight tensor(s) to bf16.", dequantized)
