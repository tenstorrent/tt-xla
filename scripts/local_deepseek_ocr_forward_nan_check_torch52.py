#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Local-only DeepSeek-OCR forward + NaN/Inf check (no Hugging Face hub load).

Use this to A/B whether **non-finite logits** on the forge path depend on the installed
``transformers`` version (e.g. dev pin ``transformers==5.2.x`` in ``venv/requirements-dev.txt``).

The hub ``AutoModel`` path is intentionally omitted: remote code and pins often disagree with
5.2.x; this script only exercises ``third_party/.../ModelLoader`` + local ``modeling_*``.

**How to use**

1. Install the ``transformers`` revision you want to test (example for repo pin)::

       pip install 'transformers==5.2.0'

2. From tt-xla repo root::

       PYTHONPATH=. python scripts/local_deepseek_ocr_forward_nan_check_torch52.py
       PYTHONPATH=. python scripts/local_deepseek_ocr_forward_nan_check_torch52.py --bfloat16

Exit code ``0`` if logits are all finite; ``1`` if any NaN/Inf appears in ``out.logits``.
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys
from pathlib import Path

import torch

from third_party.tt_forge_models.deepseek.deepseek_ocr.pytorch.loader import ModelLoader


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _transformers_version() -> str:
    return importlib.import_module("transformers").__version__


def _describe_logits(label: str, logits: torch.Tensor) -> tuple[int, int, int, bool]:
    x = logits.detach().float().cpu().flatten()
    n = int(x.numel())
    n_nan = int(torch.isnan(x).sum().item())
    n_inf = int(torch.isinf(x).sum().item())
    n_fin = int(torch.isfinite(x).sum().item())
    ok = n_nan == 0 and n_inf == 0 and n > 0
    print(
        f"{label}: shape={tuple(logits.shape)} dtype={logits.dtype} "
        f"elements={n} finite={n_fin} nan={n_nan} inf={n_inf} all_finite={ok}",
        flush=True,
    )
    return n_nan, n_inf, n_fin, ok


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--float32",
        action="store_true",
        help="Use float32 weights and activations (default).",
    )
    parser.add_argument(
        "--bfloat16",
        action="store_true",
        help="Use bfloat16 for float tensors.",
    )
    parser.add_argument(
        "--expect-transformers",
        default="5.2",
        help="Warn if installed transformers major.minor does not start with this prefix "
        "(default: 5.2). Use --expect-transformers '' to skip.",
    )
    args = parser.parse_args()
    if args.bfloat16 and args.float32:
        print("Choose at most one of --float32 / --bfloat16", file=sys.stderr)
        return 2

    tv = _transformers_version()
    print(f"transformers {tv}", flush=True)
    if args.expect_transformers and not tv.startswith(args.expect_transformers):
        print(
            f"WARNING: expected transformers prefix {args.expect_transformers!r}, "
            f"got {tv!r}. Pin e.g. pip install 'transformers==5.2.0' before concluding.",
            flush=True,
        )

    float_dtype = torch.bfloat16 if args.bfloat16 else torch.float32
    device = torch.device("cpu")

    os.chdir(_repo_root())

    loader = ModelLoader()
    print("load_inputs ...", flush=True)
    inputs = loader.load_inputs(
        dtype_override=float_dtype if args.bfloat16 else torch.float32
    )
    print("load_model ...", flush=True)
    model = loader.load_model(torch_dtype=float_dtype)
    model.eval()
    model.to(device)

    def _move_batch(batch: dict) -> dict:
        out: dict = {}
        for k, v in batch.items():
            if k == "images":
                out[k] = [
                    (
                        crop.to(device=device, dtype=float_dtype),
                        ori.to(device=device, dtype=float_dtype),
                    )
                    for crop, ori in v
                ]
            elif torch.is_tensor(v):
                if v.is_floating_point():
                    out[k] = v.to(device=device, dtype=float_dtype)
                else:
                    out[k] = v.to(device=device)
            else:
                out[k] = v
        return out

    batch = _move_batch(inputs)
    print("forward ...", flush=True)
    with torch.no_grad():
        out = model(**batch, return_dict=True, use_cache=False)

    logits = out.logits
    _n_nan, _n_inf, _n_fin, ok = _describe_logits("local forge logits", logits)

    if ok:
        print(
            "OK: logits are all finite under this transformers build "
            f"({tv!r}).",
            flush=True,
        )
        return 0
    print(
        "FAIL: non-finite values in logits under this transformers build "
        f"({tv!r}). For deeper origin, run scripts/probe_deepseek_ocr_finite_forward.py.",
        flush=True,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
