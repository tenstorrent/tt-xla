#!/usr/bin/env python3
"""Compare layer0 rotary inv_freq + RoPE matmul across transformers pins."""
from __future__ import annotations

import importlib
import os
from pathlib import Path

import torch

from third_party.tt_forge_models.deepseek.deepseek_ocr.pytorch.loader import ModelLoader


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _report_inv_freq_and_matmul(model) -> None:
    rotary = model.model.layers[0].rotary_emb
    inv = rotary.inv_freq.detach().float().cpu()
    print(f"  rope_type={getattr(rotary, 'rope_type', '?')}")
    print(f"  inv_freq: shape={tuple(inv.shape)} min={inv.min().item():.6g} max={inv.max().item():.6g}")
    print(f"  inv_freq finite={bool(torch.isfinite(inv).all())} nan={int(torch.isnan(inv).sum())} inf={int(torch.isinf(inv).sum())}")

    pos = torch.arange(913, dtype=torch.int64).view(1, -1)
    inv_exp = inv[None, :, None].expand(1, -1, 1)
    pos_exp = pos[:, None, :].float()
    freqs = (inv_exp @ pos_exp).transpose(1, 2)
    print(f"  freqs@pos[0:912]: shape={tuple(freqs.shape)} min={freqs.min().item():.6g} max={freqs.max().item():.6g}")
    print(f"  freqs finite={bool(torch.isfinite(freqs).all())} nan={int(torch.isnan(freqs).sum())} inf={int(torch.isinf(freqs).sum())}")

    cfg = model.model.layers[0].rotary_emb.config
    rp = getattr(cfg, "rope_parameters", None)
    print(f"  config.rope_theta={getattr(cfg, 'rope_theta', None)}")
    print(f"  config.rope_parameters={rp}")


def main() -> int:
    os.chdir(_repo_root())
    tv = importlib.import_module("transformers").__version__
    print(f"transformers {tv}", flush=True)

    loader = ModelLoader()
    model = loader.load_model(torch_dtype=torch.float32)
    model.eval()
    _report_inv_freq_and_matmul(model)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
