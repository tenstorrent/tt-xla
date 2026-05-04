# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
CPU correctness test: original vs optimised masked_scatter decomposition.

Shapes from DeepSeek OCR CPU run:
  data   (inputs_embeds[idx])         : [913, 1280]  bfloat16
  mask   (images_seq_mask unsqueezed) : [913, 1]     bool
  source (images_in_this_batch)       : [903, 1280]  bfloat16
"""

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


def test_deepseek_ocr_masked_scatter_cpu():
    torch.manual_seed(0)

    data = torch.randn(913, 1280, dtype=torch.bfloat16)
    source = torch.randn(903, 1280, dtype=torch.bfloat16)
    mask = torch.zeros(913, 1, dtype=torch.bool)
    mask[:903, :] = True

    out_orig = masked_scatter_original(data.clone(), mask, source)
    out_opt = masked_scatter_optimised(data.clone(), mask, source)

    match = torch.equal(out_orig, out_opt)
    pcc = compute_pcc(out_orig, out_opt)

    print(f"torch.equal : {match}")
    print(f"PCC         : {pcc:.6f}")
    assert match, f"Outputs differ! PCC={pcc:.6f}"
