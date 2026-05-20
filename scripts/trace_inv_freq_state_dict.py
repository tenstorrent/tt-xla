#!/usr/bin/env python3
"""Find which checkpoint key corrupts rotary_emb.inv_freq on transformers 5.2."""
from __future__ import annotations

import os
from pathlib import Path

import torch
from safetensors.torch import load_file

from third_party.tt_forge_models.deepseek.deepseek_ocr.pytorch.src.modeling_deepseekocr import (
    DeepseekOCRConfig,
    DeepseekOCRForCausalLM,
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _inv_stats(model, tag: str) -> None:
    inv = model.model.layers[0].rotary_emb.inv_freq.detach().float().cpu()
    print(f"{tag}: min={inv.min().item():.6g} max={inv.max().item():.6g}", flush=True)


def main() -> int:
    os.chdir(_repo_root())
    path = "DeepSeek_OCR_weights"
    cfg = DeepseekOCRConfig.from_pretrained(path, local_files_only=True)
    model = DeepseekOCRForCausalLM(cfg)
    _inv_stats(model, "before load")

    sd = load_file(f"{path}/model-00001-of-000001.safetensors", device="cpu")
    target = "model.layers.0.rotary_emb.inv_freq"
    if target in sd:
        print(f"checkpoint contains {target}", flush=True)
    else:
        print(f"checkpoint has no {target}", flush=True)

    # Keys that might collide with inv_freq buffer name
    for k in sorted(sd.keys()):
        if "inv_freq" in k or "rotary" in k:
            t = sd[k]
            print(f"  ckpt key {k}: shape={tuple(t.shape)}", flush=True)

    missing, unexpected = model.load_state_dict(sd, strict=False)
    _inv_stats(model, "after load_state_dict")

    rotary_keys = [k for k in model.state_dict() if "rotary" in k or "inv_freq" in k]
    print(f"model rotary-related keys ({len(rotary_keys)}):", flush=True)
    for k in rotary_keys[:20]:
        t = model.state_dict()[k]
        print(f"  {k}: shape={tuple(t.shape)}", flush=True)

    if missing:
        print(f"missing keys count={len(missing)} (first 5): {missing[:5]}", flush=True)
    if unexpected:
        print(f"unexpected keys count={len(unexpected)} (first 10): {unexpected[:10]}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
