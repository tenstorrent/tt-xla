#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Re-run only ``model.layers[0].rotary_emb`` on the **same** tensors used in a full forward.

After one full ``DeepseekOCRForCausalLM`` forward, this captures:

- ``hidden_after_input_layernorm`` (output of ``layers[0].input_layernorm``)
- ``position_ids`` (from the kwargs passed into ``layers[0]``)

Then it calls ``rotary_emb(hidden, position_ids)`` again in ``torch.no_grad()`` and reports whether
cos/sin are finite. Optionally mirrors Hugging Face ``LlamaRotaryEmbedding.forward`` math step-by-step
(``--rope-steps``) to see which intermediate first loses finiteness.

Run from tt-xla repo root::

    PYTHONPATH=. python scripts/test_deepseek_ocr_layer0_rotary_standalone.py
    PYTHONPATH=. python scripts/test_deepseek_ocr_layer0_rotary_standalone.py --device cuda --rope-steps
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from third_party.tt_forge_models.deepseek.deepseek_ocr.pytorch.loader import ModelLoader


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _move_inputs(batch: dict, device: torch.device, float_dtype: torch.dtype) -> dict:
    out: dict = {}
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


def _finite_line(tag: str, t: torch.Tensor) -> str:
    n = t.numel()
    if n == 0:
        return f"{tag}: empty"
    ok = bool(torch.isfinite(t).all().item())
    n_nan = int(torch.isnan(t).sum().item())
    n_inf = int(torch.isinf(t).sum().item())
    return f"{tag}: all_finite={ok} shape={tuple(t.shape)} dtype={t.dtype} nan={n_nan} inf={n_inf}"


def _rope_steps_like_hf(rotary: torch.nn.Module, x: torch.Tensor, position_ids: torch.Tensor) -> None:
    """Mirror transformers LlamaRotaryEmbedding.forward lines ~125-133 (float32 path)."""
    inv_freq = rotary.inv_freq
    attention_scaling = float(getattr(rotary, "attention_scaling", 1.0))
    print("--- RoPE step mirror (HF-style, float32 intermediates) ---", flush=True)
    print(_finite_line("inv_freq buffer", inv_freq), flush=True)

    inv_freq_expanded = inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
    position_ids_expanded = position_ids[:, None, :].float()
    print(_finite_line("inv_freq_expanded", inv_freq_expanded), flush=True)
    print(_finite_line("position_ids_expanded", position_ids_expanded), flush=True)

    freqs = (inv_freq_expanded @ position_ids_expanded).transpose(1, 2)
    print(_finite_line("freqs (matmul)", freqs), flush=True)

    emb = torch.cat((freqs, freqs), dim=-1)
    print(_finite_line("emb (cat)", emb), flush=True)

    cos = emb.cos() * attention_scaling
    sin = emb.sin() * attention_scaling
    print(_finite_line("cos (before cast to x.dtype)", cos), flush=True)
    print(_finite_line("sin (before cast to x.dtype)", sin), flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--bfloat16", action="store_true")
    parser.add_argument(
        "--rope-steps",
        action="store_true",
        help="Print finiteness after each RoPE intermediate (HF forward mirror).",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    float_dtype = torch.bfloat16 if args.bfloat16 else torch.float32

    os.chdir(_repo_root())

    caps: Dict[str, torch.Tensor] = {}
    handles: List[Any] = []

    loader = ModelLoader()
    inputs = loader.load_inputs(
        dtype_override=float_dtype if args.bfloat16 else torch.float32
    )
    model = loader.load_model(torch_dtype=float_dtype)
    model.eval()
    model.to(device)
    batch = _move_inputs(inputs, device, float_dtype)

    layer0 = model.model.layers[0]
    rotary = layer0.rotary_emb

    def pre_layer0(_mod: torch.nn.Module, args: Any, kwargs: Any = None) -> None:
        kwargs = kwargs or {}
        pos = kwargs.get("position_ids")
        if pos is not None and "position_ids" not in caps:
            caps["position_ids"] = pos.detach().clone()

    def hook_input_ln(_mod: torch.nn.Module, _inp: Any, out: Any) -> None:
        if "hidden_after_input_ln" not in caps:
            caps["hidden_after_input_ln"] = out.detach().clone()

    def _register_pre_with_kwargs(mod: torch.nn.Module, hook) -> Any:
        try:
            return mod.register_forward_pre_hook(hook, with_kwargs=True)
        except TypeError:
            def shim(m, a):
                return hook(m, a, {})

            return mod.register_forward_pre_hook(shim)

    handles.append(_register_pre_with_kwargs(layer0, pre_layer0))
    handles.append(layer0.input_layernorm.register_forward_hook(hook_input_ln))

    print("Running one full forward to capture layer0 inputs to rotary_emb ...", flush=True)
    with torch.no_grad():
        model(**batch, return_dict=True, use_cache=False)

    for h in handles:
        h.remove()

    if "hidden_after_input_ln" not in caps or "position_ids" not in caps:
        print(
            "ERROR: capture failed (missing hidden_after_input_ln or position_ids).",
            flush=True,
        )
        return 2

    x = caps["hidden_after_input_ln"].to(device=device, dtype=float_dtype)
    pos = caps["position_ids"].to(device=device)
    print("--- Captured tensors ---", flush=True)
    print(_finite_line("hidden_after_input_ln", x), flush=True)
    print(f"position_ids: shape={tuple(pos.shape)} dtype={pos.dtype} min={pos.min().item()} max={pos.max().item()}", flush=True)

    if args.rope_steps:
        _rope_steps_like_hf(rotary, x, pos)

    print("--- Standalone rotary_emb(x, position_ids) ---", flush=True)
    with torch.no_grad():
        out = rotary(x, pos)
    if not isinstance(out, tuple) or len(out) != 2:
        print(f"Unexpected rotary output type: {type(out)}", flush=True)
        return 3
    cos, sin = out
    print(_finite_line("cos", cos), flush=True)
    print(_finite_line("sin", sin), flush=True)

    ok = bool(torch.isfinite(cos).all().item() and torch.isfinite(sin).all().item())
    if ok:
        print("RESULT: rotary_emb alone is finite on replayed tensors.", flush=True)
        return 0
    print("RESULT: rotary_emb reproduces non-finite cos/sin on replayed tensors.", flush=True)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
