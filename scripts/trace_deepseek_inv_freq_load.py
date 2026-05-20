#!/usr/bin/env python3
"""Trace when layer0.rotary_emb.inv_freq becomes corrupted during load (transformers 5.2)."""
from __future__ import annotations

import importlib
import os
from pathlib import Path

import torch

from third_party.tt_forge_models.deepseek.deepseek_ocr.pytorch.src.modeling_deepseekocr import (
    DeepseekOCRConfig,
    DeepseekOCRForCausalLM,
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _inv_stats(model, tag: str) -> None:
    inv = model.model.layers[0].rotary_emb.inv_freq.detach().float().cpu()
    print(
        f"{tag}: min={inv.min().item():.6g} max={inv.max().item():.6g} "
        f"finite={bool(torch.isfinite(inv).all())}",
        flush=True,
    )


def main() -> int:
    os.chdir(_repo_root())
    print(f"transformers {importlib.import_module('transformers').__version__}", flush=True)

    path = "DeepSeek_OCR_weights"
    cfg = DeepseekOCRConfig.from_pretrained(path, local_files_only=True)
    print("DeepseekOCRForCausalLM(config) only ...", flush=True)
    model = DeepseekOCRForCausalLM(cfg)
    _inv_stats(model, "after __init__(config)")

    print("from_pretrained (full) ...", flush=True)
    model2 = DeepseekOCRForCausalLM.from_pretrained(
        path,
        local_files_only=True,
        trust_remote_code=True,
        torch_dtype=torch.float32,
    )
    _inv_stats(model2, "after from_pretrained")

    model2.eval()
    _inv_stats(model2, "after eval")

    model2.to(torch.float32)
    _inv_stats(model2, "after to(float32)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
