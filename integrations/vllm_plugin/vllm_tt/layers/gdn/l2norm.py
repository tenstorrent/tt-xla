# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Portions (c) 2026 Tenstorrent AI ULC
"""Pure-PyTorch L2 normalization, matching ``fla...ops.l2norm.l2norm_fwd``."""

import torch


def tt_l2norm_fwd(
    x: torch.Tensor,
    eps: float = 1e-6,
    output_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """L2-normalize over the last dim: ``x / sqrt(sum(x**2, -1) + eps)``.

    Mirrors FLA's ``l2norm_fwd`` semantics, including the ``+ eps`` *inside* the
    square root and the float32 reduction. The default ``eps`` matches FLA.
    """
    x_dtype = x.dtype if output_dtype is None else output_dtype
    xf = x.to(torch.float32)
    var = xf.pow(2).sum(dim=-1, keepdim=True)
    y = xf * torch.rsqrt(var + eps)
    return y.to(x_dtype)
