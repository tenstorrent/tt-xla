#!/usr/bin/env python3
"""Isolate RoPE matmul: inv_freq @ position_ids — same math as LlamaRotaryEmbedding.forward."""
from __future__ import annotations

import importlib
import os
from pathlib import Path

import torch
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding

from third_party.tt_forge_models.deepseek.deepseek_ocr.pytorch.loader import ModelLoader
from third_party.tt_forge_models.deepseek.deepseek_ocr.pytorch.src.configuration_deepseek_v2 import (
    DeepseekV2Config,
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def main() -> int:
    os.chdir(_repo_root())
    tv = importlib.import_module("transformers").__version__
    print(f"transformers {tv}", flush=True)

    cfg_path = "DeepSeek_OCR_weights/config.json"
    cfg = DeepseekV2Config.from_pretrained(cfg_path)
    print(f"[config-only] rope_theta={cfg.rope_theta}", flush=True)
    rp = getattr(cfg, "rope_parameters", None)
    print(f"[config-only] rope_parameters={rp}", flush=True)

    rotary = LlamaRotaryEmbedding(config=cfg)
    inv = rotary.inv_freq.detach().float()
    print(f"inv_freq shape={tuple(inv.shape)} min={inv.min().item():.6g} max={inv.max().item():.6g}", flush=True)
    print(
        f"inv_freq finite={bool(torch.isfinite(inv).all())} nan={int(torch.isnan(inv).sum())} "
        f"inf={int(torch.isinf(inv).sum())}",
        flush=True,
    )

    pos = torch.arange(913, dtype=torch.int64).view(1, -1)
    inv_exp = inv[None, :, None].expand(1, -1, 1)
    pos_exp = pos[:, None, :].float()
    freqs = (inv_exp @ pos_exp).transpose(1, 2)
    print(f"freqs shape={tuple(freqs.shape)} min={freqs.min().item():.6g} max={freqs.max().item():.6g}", flush=True)
    print(
        f"freqs finite={bool(torch.isfinite(freqs).all())} nan={int(torch.isnan(freqs).sum())} "
        f"inf={int(torch.isinf(freqs).sum())}",
        flush=True,
    )

    x = torch.zeros(1, 913, 1280)
    cos, sin = rotary(x, pos)
    print(
        f"[config-only] cos finite={bool(torch.isfinite(cos).all())} nan={int(torch.isnan(cos).sum())} "
        f"inf={int(torch.isinf(cos).sum())}",
        flush=True,
    )

    print("--- full ModelLoader model.layers[0].rotary_emb ---", flush=True)
    model = ModelLoader().load_model(torch_dtype=torch.float32)
    model.eval()
    layer_cfg = model.model.layers[0].rotary_emb.config
    print(f"[loaded] layer rotary config type={type(layer_cfg).__name__}", flush=True)
    print(f"[loaded] rope_theta={getattr(layer_cfg, 'rope_theta', None)}", flush=True)
    print(f"[loaded] rope_parameters={getattr(layer_cfg, 'rope_parameters', None)}", flush=True)
    inv2 = model.model.layers[0].rotary_emb.inv_freq.detach().float()
    print(f"[loaded] inv_freq shape={tuple(inv2.shape)} min={inv2.min().item():.6g} max={inv2.max().item():.6g}", flush=True)
    print(
        f"[loaded] inv_freq finite={bool(torch.isfinite(inv2).all())} nan={int(torch.isnan(inv2).sum())} "
        f"inf={int(torch.isinf(inv2).sum())}",
        flush=True,
    )
    freqs2 = (inv2[None, :, None] @ pos[:, None, :].float()).transpose(1, 2)
    print(f"[loaded] freqs min={freqs2.min().item():.6g} max={freqs2.max().item():.6g}", flush=True)
    print(
        f"[loaded] freqs finite={bool(torch.isfinite(freqs2).all())} nan={int(torch.isnan(freqs2).sum())} "
        f"inf={int(torch.isinf(freqs2).sum())}",
        flush=True,
    )
    return 0 if torch.isfinite(freqs2).all() else 1


if __name__ == "__main__":
    raise SystemExit(main())
