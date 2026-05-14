#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Compare DeepSeek-OCR logits: Hugging Face hub (remote code) vs tt-forge ModelLoader (local modeling).

Use this to confirm CPU reference parity between the upstream-style load path and the path used
by tt-xla tests before attributing PCC failures to the TT/XLA run.

Prerequisites (same as model tests):
  - HF Hub access for ``deepseek-ai/DeepSeek-OCR`` (optional HF_TOKEN for rate limits).
  - ``test_images/doc.png`` resolvable via tt_forge_models ``get_file`` (e.g. IRD_LF_CACHE / cache).

Example (from tt-xla repo root; ``third_party`` imports need the repo on ``PYTHONPATH``,
same as tests and ``tests/benchmark/README.md``)::

    PYTHONPATH=. python scripts/compare_deepseek_ocr_hf_vs_forge_loader.py
    PYTHONPATH=. python scripts/compare_deepseek_ocr_hf_vs_forge_loader.py --device cuda
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
from pathlib import Path

from third_party.tt_forge_models.deepseek.deepseek_ocr.pytorch.loader import ModelLoader


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _compute_pcc(a: "torch.Tensor", b: "torch.Tensor") -> float:
    import torch

    x = a.to(torch.float32).flatten()
    y = b.to(torch.float32).flatten()
    xc = x - x.mean()
    yc = y - y.mean()
    denom = xc.norm() * yc.norm()
    if denom == 0:
        if torch.allclose(x, y, rtol=1e-2, atol=1e-2):
            return 1.0
        return float("nan")
    pcc = ((xc @ yc) / denom).item()
    return max(-1.0, min(1.0, pcc))


def _move_inputs(batch: dict, device: "torch.device", float_dtype: "torch.dtype") -> dict:
    import torch

    out = {}
    for k, v in batch.items():
        if k == "images":
            moved = []
            for crop, ori in v:
                moved.append(
                    (
                        crop.to(device=device, dtype=float_dtype),
                        ori.to(device=device, dtype=float_dtype),
                    )
                )
            out[k] = moved
        elif torch.is_tensor(v):
            if v.is_floating_point():
                out[k] = v.to(device=device, dtype=float_dtype)
            else:
                out[k] = v.to(device=device)
        else:
            out[k] = v
    return out


def _forward_logits(model, batch: dict, device: "torch.device", float_dtype) -> "torch.Tensor":
    import torch

    batch = _move_inputs(batch, device, float_dtype)
    with torch.no_grad():
        out = model(**batch, return_dict=True, use_cache=False)
    logits = out.logits
    if torch.isnan(logits).any():
        print("WARNING: logits contain NaN", flush=True)
    return logits.detach().float().cpu()


def main() -> int:
    import torch
    from transformers import AutoModelForCausalLM

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--hf-repo",
        default="deepseek-ai/DeepSeek-OCR",
        help="Hub repo id for the reference (remote-code) model.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device for inference (cpu or cuda). Large weights: cuda recommended.",
    )
    parser.add_argument(
        "--no-sequential",
        action="store_true",
        help="Keep both models in memory (higher peak RAM; default loads one model at a time).",
    )
    parser.add_argument(
        "--float32",
        action="store_true",
        help="Use float32 weights and inputs (default).",
    )
    parser.add_argument(
        "--bfloat16",
        action="store_true",
        help="Use bfloat16 for float tensors (may differ slightly from float32 reference).",
    )
    args = parser.parse_args()
    if args.bfloat16 and args.float32:
        print("Choose at most one of --float32 / --bfloat16", file=sys.stderr)
        return 2

    sequential = not args.no_sequential
    float_dtype = torch.bfloat16 if args.bfloat16 else torch.float32
    device = torch.device(args.device)

    os.chdir(_repo_root())

    print("Building inputs via forge ModelLoader.load_inputs() ...", flush=True)
    loader = ModelLoader()
    inputs = loader.load_inputs(dtype_override=float_dtype if args.bfloat16 else torch.float32)

    print("Loading forge (local modeling + snapshot weights) ...", flush=True)
    forge_model = loader.load_model(torch_dtype=float_dtype)
    forge_model.eval()
    forge_model.to(device)
    logits_forge = _forward_logits(forge_model, inputs, device, float_dtype)
    if sequential:
        del forge_model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print(f"Loading HF AutoModelForCausalLM from {args.hf_repo!r} (trust_remote_code=True) ...", flush=True)
    hf_model = AutoModelForCausalLM.from_pretrained(
        args.hf_repo,
        trust_remote_code=True,
        torch_dtype=float_dtype,
    )
    hf_model.eval()
    hf_model.to(device)
    logits_hf = _forward_logits(hf_model, inputs, device, float_dtype)

    if not sequential:
        del forge_model
    del hf_model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    if logits_forge.shape != logits_hf.shape:
        print(
            f"MISMATCH shapes: forge {tuple(logits_forge.shape)} vs hf {tuple(logits_hf.shape)}",
            file=sys.stderr,
        )
        return 1

    max_abs = (logits_forge - logits_hf).abs().max().item()
    pcc = _compute_pcc(logits_forge, logits_hf)

    print("---", flush=True)
    print(f"logits shape: {tuple(logits_forge.shape)}", flush=True)
    print(f"max_abs_diff: {max_abs}", flush=True)
    print(f"pcc (forge vs hf): {pcc}", flush=True)
    if pcc == pcc and pcc >= 0.999 and max_abs < 1e-3:
        print("OK: forge loader path matches HF hub path closely on CPU.", flush=True)
        return 0
    if pcc == pcc and pcc >= 0.99:
        print("CLOSE: PCC >= 0.99; small drift may be dtype/nondeterminism.", flush=True)
        return 0
    print(
        "DIVERGED: local forge modeling or weights path differs from hub remote code "
        "(investigate before blaming TT/XLA).",
        flush=True,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
