# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Host-side BFP4/BFP8 packing for pre-quantized weight ship.

Bypasses the on-device const_eval `bf16 → bfp_bf4/bf8` typecast (which
keeps both bf16 input and bfp4 cache resident in DRAM, OOM'ing for
DeepSeek-V4-Pro). Instead, packs on host, ships packed bytes via a U8
torch_xla tensor, and the compiler/plugin reinterpret the bytes as a
block-float tile-layout tensor on the device side — no bf16 ever
materializes in device DRAM.

Pipeline:
    bf16 weight (CPU)
      → pack_to_bfp4(...) using ttnn._ttnn.bfp_utils.pack_bfp4   [HOST]
      → torch.uint8 [TT bytes]
      → torch.ops.tt.weight_pre_quantized(packed, "bfp_bf4",
                                          logical_shape)         [TRACE]
      → stablehlo.custom_call @tt.weight_pre_quantized
      → (downstream) compiler pass re-types the source arg to a
        BFP4-tiled tensor; plugin ships U8 bytes as BFP_BFloat4 host
        runtime tensor.

A single 32x32 tile in BFP_BFloat4_B is laid out as:
    [16 byte exponents (one per row-half)] +
    [16 * 16 * 0.5 = 128 byte mantissas (4-bit nibbles, packed)]
Total per tile = 16 + 128 = 144 bytes (vs 32 * 32 * 2 = 2048 bytes for bf16).

The exact byte layout matches what `ttnn.from_torch(t,
dtype=ttnn.bfloat4_b, layout=ttnn.TILE_LAYOUT)` produces; we use
`ttnn._ttnn.bfp_utils.pack_bfp4` directly to skip the heavyweight
ttnn.Tensor wrapper, which would otherwise grab the device.
"""
from __future__ import annotations

from typing import List, Sequence

import numpy as np
import torch


_TILE = 32


def _lazy_pack_fn(mant_bits: int):
    """Resolve `ttnn._ttnn.bfp_utils.pack_bfpN` lazily.

    Importing `ttnn` at module load opens the device cluster, which
    conflicts with PJRT's mesh device. We import inside the call so the
    tt-xla path still works.
    """
    import ttnn  # noqa: WPS433

    fn = {
        3: ttnn._ttnn.bfp_utils.pack_bfp4,  # 1 sign + 3 mantissa bits
        7: ttnn._ttnn.bfp_utils.pack_bfp8,  # 1 sign + 7 mantissa bits
    }
    if mant_bits not in fn:
        raise ValueError(
            f"Unsupported mant_bits={mant_bits}; valid: {list(fn.keys())}"
        )
    return fn[mant_bits]


def _dtype_to_mant_bits(dtype_str: str) -> int:
    """Map a tt-mlir dtype string to the corresponding mantissa width."""
    return {"bfp_bf4": 3, "bfp_bf8": 7}[dtype_str]


def pack_to_bfp(
    weight: torch.Tensor, dtype_str: str = "bfp_bf4"
) -> torch.Tensor:
    """Pack a 2D bf16/fp32 weight on host into raw BFP_BFloat4/BF8 bytes.

    Returns a 1D `torch.uint8` tensor whose storage matches the on-device
    block-float tile layout for the given dtype. Pads each dim up to a
    multiple of 32 (tile size) — the padded values are zero, ignored at
    matmul time after we re-type the arg to its logical (un-padded) shape
    in the compiler.

    Args:
        weight: 2D tensor (any float dtype, will be cast to fp32 for packing).
        dtype_str: "bfp_bf4" or "bfp_bf8".

    Returns:
        torch.uint8 1D tensor of total packed bytes.
    """
    if weight.dim() != 2:
        raise ValueError(
            f"pack_to_bfp expects 2D tensor, got shape {tuple(weight.shape)}"
        )
    mant_bits = _dtype_to_mant_bits(dtype_str)
    pack_fn = _lazy_pack_fn(mant_bits)

    h, w = weight.shape
    h_pad = ((h + _TILE - 1) // _TILE) * _TILE
    w_pad = ((w + _TILE - 1) // _TILE) * _TILE

    padded = np.zeros((h_pad, w_pad), dtype=np.float32)
    padded[:h, :w] = weight.detach().to(torch.float32).cpu().numpy()

    # `pack_bfpN` operates on a contiguous block of tiles laid out
    # row-major. Pass the entire padded tensor in one call — the
    # implementation walks tile boundaries internally (matches what
    # `ttnn.from_torch(..., layout=TILE_LAYOUT)` produces).
    packed_u32 = np.asarray(pack_fn(padded.ravel(), row_major_input=True))
    packed_u8 = packed_u32.view(np.uint8)
    return torch.from_numpy(packed_u8.copy())


# ---- Custom op marker --------------------------------------------------


@torch.library.custom_op(
    "tt::weight_pre_quantized", mutates_args=[], device_types=["cpu", "xla"]
)
def weight_pre_quantized(
    packed: torch.Tensor, dtype_str: str, logical_shape: List[int]
) -> torch.Tensor:
    """Marker op that wraps host-pre-packed BFP4/BFP8 bytes.

    On XLA, traces to `stablehlo.custom_call @tt.weight_pre_quantized`
    with the packed U8 input and frontend_attributes carrying
    `ttcore.weight_dtype` and `ttcore.weight_pre_quantized_logical_shape`.
    A tt-xla frontend pass consumes the custom_call: re-types the source
    function argument to `tensor<logical_shape × bf16>` with a BFP_BFloat4
    tile layout encoding, and removes the custom_call. The plugin's
    `copyFromHostBuffer` reads the same metadata to build a BFP4 host
    runtime tensor over the U8 bytes.

    The eager (CPU) implementation just returns a zeros tensor of the
    logical shape — never run on the hot path; only used so dynamo
    fake-tensor tracing has a shape/dtype to propagate.

    Args:
        packed: 1D `torch.uint8` tensor of packed bytes.
        dtype_str: "bfp_bf4" or "bfp_bf8".
        logical_shape: shape of the LOGICAL bf16 tensor (un-padded). The
            compiler uses this to size the re-typed arg.

    Returns:
        A bf16 tensor of `logical_shape`.
    """
    if packed.device.type == "cpu":
        return torch.zeros(logical_shape, dtype=torch.bfloat16)

    assert dtype_str in ("bfp_bf4", "bfp_bf8"), (
        f"weight_pre_quantized: dtype_str must be 'bfp_bf4' or 'bfp_bf8', "
        f"got {dtype_str!r}"
    )
    assert packed.dtype == torch.uint8, (
        f"weight_pre_quantized: packed must be uint8, got {packed.dtype}"
    )

    # stablehlo_custom_call runs into XLA shape-inference issues for 1D
    # or 2D outputs in some configurations (mirror of the workaround in
    # weight_dtype_override). Reshape the LOGICAL output to 3D for the
    # custom_call, then reshape back. The custom_call's INPUT remains 1D
    # (uint8 raw bytes) — only the output's logical shape is touched.
    from torch_xla.experimental import stablehlo_custom_call  # noqa: WPS433

    output_shape = list(logical_shape)
    extra_dims: List[int] = []
    if len(output_shape) < 3:
        extra_dims = [1] * (3 - len(output_shape))
        traced_output_shape = (*extra_dims, *output_shape)
    else:
        traced_output_shape = tuple(output_shape)

    frontend_attributes = {
        "ttcore.weight_dtype": dtype_str,
        "ttcore.weight_pre_quantized_logical_shape": ",".join(
            str(d) for d in logical_shape
        ),
    }

    result = stablehlo_custom_call.stablehlo_custom_call(
        [packed],
        "tt.weight_pre_quantized",
        [traced_output_shape],
        [torch.bfloat16],
        frontend_attributes=frontend_attributes,
    )

    if extra_dims:
        result = result.reshape(output_shape)
    return result


@weight_pre_quantized.register_fake
def _(
    packed: torch.Tensor, dtype_str: str, logical_shape: List[int]
) -> torch.Tensor:
    """FakeTensor impl for dynamo. Just emit a zeros tensor of logical shape."""
    return torch.zeros(logical_shape, dtype=torch.bfloat16)


@weight_pre_quantized.register_autograd
def _(ctx, grad_output):
    """No autograd needed for inference-only weights."""
    return grad_output, None, None
