# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
CPU correctness test: original vs optimised masked_scatter decomposition.

Shapes from Gemma3 CPU runs:
  4b:  data [1, 277, 2560] bfloat16 | mask [1, 277, 2560] bool | source [1, 256, 2560] bfloat16
  12b: data [1, 277, 3840] bfloat16 | mask [1, 277, 3840] bool | source [1, 256, 3840] bfloat16
"""

import pytest
import torch


def masked_scatter_original(
    data: torch.Tensor, mask: torch.Tensor, source: torch.Tensor
) -> torch.Tensor:
    mask, data = torch.broadcast_tensors(mask, data)
    mask_f = mask.reshape(-1)
    data_flat = data.reshape(-1)
    source_flat = source.reshape(-1)
    mask_i = mask_f.long()
    source_idx = torch.cumsum(mask_i, 0) - 1
    source_idx = torch.clamp(source_idx, 0, source_flat.numel() - 1)
    gathered = source_flat[source_idx]
    result_flat = torch.where(mask_f, gathered, data_flat)
    return result_flat.view_as(data)


def masked_scatter_optimised(
    data: torch.Tensor, mask: torch.Tensor, source: torch.Tensor
) -> torch.Tensor:
    H = data.shape[-1] if data.ndim >= 2 else 0
    n_true = source.numel() // H if H > 0 else 0
    _row_constant = (
        data.ndim >= 2
        and mask.ndim >= 2
        and (
            mask.stride(-1) == 0
            or mask.ndim < data.ndim
            or (n_true > 0 and source.numel() == n_true * H)
        )
    )

    mask, data = torch.broadcast_tensors(mask, data)

    if _row_constant and n_true > 0:
        mask_outer = mask[..., 0]
        mask_f = mask_outer.reshape(-1)
        data_2d = data.reshape(-1, H)
        source_2d = source.reshape(n_true, H)
        mask_i = mask_f.long()
        source_idx = torch.cumsum(mask_i, 0) - 1
        source_idx = torch.clamp(source_idx, 0, n_true - 1)
        gathered = source_2d[source_idx]
        result = torch.where(mask_f.unsqueeze(-1), gathered, data_2d)
        return result.view_as(data)

    mask_f = mask.reshape(-1)
    data_flat = data.reshape(-1)
    source_flat = source.reshape(-1)
    mask_i = mask_f.long()
    source_idx = torch.cumsum(mask_i, 0) - 1
    source_idx = torch.clamp(source_idx, 0, source_flat.numel() - 1)
    gathered = source_flat[source_idx]
    result_flat = torch.where(mask_f, gathered, data_flat)
    return result_flat.view_as(data)


def compute_pcc(x: torch.Tensor, y: torch.Tensor) -> float:
    x_flat = x.float().flatten()
    y_flat = y.float().flatten()
    vx = x_flat - x_flat.mean()
    vy = y_flat - y_flat.mean()
    denom = vx.norm() * vy.norm()
    if denom == 0:
        return float("nan")
    return ((vx @ vy) / denom).item()


@pytest.mark.parametrize(
    "variant,H,n_true",
    [
        ("4b",  2560, 256),
        ("12b", 3840, 256),
    ],
)
def test_gemma3_masked_scatter_cpu(variant, H, n_true):
    torch.manual_seed(0)

    # Exact shapes from Gemma3 CPU runs (common: seq_len=277, batch=1)
    data = torch.randn(1, 277, H, dtype=torch.bfloat16)
    source = torch.randn(1, n_true, H, dtype=torch.bfloat16)

    # Row-constant mask: 256 out of 277 token positions are image tokens
    mask = torch.zeros(1, 277, H, dtype=torch.bool)
    mask[0, :n_true, :] = True

    out_orig = masked_scatter_original(data.clone(), mask, source)
    out_opt = masked_scatter_optimised(data.clone(), mask, source)

    match = torch.equal(out_orig, out_opt)
    pcc = compute_pcc(out_orig, out_opt)

    print(f"[gemma3-{variant}] torch.equal: {match}  PCC: {pcc:.6f}")
    assert match, f"[gemma3-{variant}] Outputs differ! PCC={pcc:.6f}"
