# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Portions (c) 2026 Tenstorrent AI ULC
"""Pure-PyTorch GDN gating, matching ``fused_gdn_gating`` in vLLM's GDN layer.

``g = -exp(A_log.float()) * softplus(a.float() + dt_bias)``  (log-decay)
``beta = sigmoid(b.float())``                                (write strength)
"""

import torch
import torch.nn.functional as F


def tt_fused_gdn_gating(
    A_log: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    dt_bias: torch.Tensor,
    beta: float = 1.0,
    threshold: float = 20.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the GDN log-decay ``g`` and write strength ``beta_out``.

    Args:
        A_log: per-head log of the base decay, shape ``[HV]`` (float32).
        a: gating pre-activation, shape ``[..., HV]``.
        b: write-strength pre-activation, shape ``[..., HV]``.
        dt_bias: per-head time-step bias, shape ``[HV]``.
        beta, threshold: softplus stability params (match FLA's Triton kernel).

    Returns:
        ``(g, beta_out)``, both float32 with the same shape as ``a``/``b``.
        ``g`` is log-space decay (exp'd at use-site).
    """
    # F.softplus with (beta, threshold) reproduces the kernel's stable branch:
    #   beta*x <= threshold -> (1/beta)*log(1+exp(beta*x)); else x.
    x = a.to(torch.float32) + dt_bias.to(torch.float32)
    softplus_x = F.softplus(x, beta=beta, threshold=threshold)
    g = -torch.exp(A_log.to(torch.float32)) * softplus_x
    beta_out = torch.sigmoid(b.to(torch.float32))
    return g, beta_out
