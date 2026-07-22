# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Shared helper for clamping out-of-range negative slice bounds (tt-xla #4465).

CPU eager clamps a slice bound ``< -dim_size`` to ``-dim_size``; XLA rejects it.
Reused by the eager ``__getitem__`` override and the pre-AOTAutograd getitem pass.
"""
import torch


def clamp_neg_slice_key(shape: torch.Size, key):
    """Clamp negative slice starts/ends ``< -dim_size`` in a ``__getitem__`` key.

    Returns ``(new_key, changed)``. Bails on Ellipsis; leaves symbolic dims and
    non-slice indices untouched.
    """
    key_tuple = key if isinstance(key, tuple) else (key,)
    if any(k is Ellipsis for k in key_tuple):
        return key, False

    changed = False
    out = []
    dim = 0
    for k in key_tuple:
        if k is None:  # newaxis inserts a dim, consumes none of ``shape``
            out.append(k)
            continue
        if isinstance(k, slice) and dim < len(shape):
            size = shape[dim]
            if isinstance(size, int):
                start, stop, step = k.start, k.stop, k.step
                if isinstance(start, int) and start < -size:
                    start = -size
                    changed = True
                if isinstance(stop, int) and stop < -size:
                    stop = -size
                    changed = True
                k = slice(start, stop, step)
        dim += 1  # int / slice / tensor index each consume one dim
        out.append(k)

    if not changed:
        return key, False
    return (tuple(out) if isinstance(key, tuple) else out[0]), True
